#!/usr/bin/env python

import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tqdm
from pydantic import FilePath

import modalities.inference.inference as inference
from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.instantiation_models import PackedDatasetComponentsInstantiationModel
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.dataloader.preprocessing.tokenization.tokenized_file_writer import TokenizedFileWriter
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.preprocessing.create_chunks import Chunking
from modalities.preprocessing.shuffle_data import DataShuffler
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.utils.logging import get_logger
from modalities.utils.seeding import calculate_hashed_seed


class FileExistencePolicy(Enum):
    SKIP = "skip"
    ERROR = "error"
    OVERRIDE = "override"


def enforce_file_existence_policy(file_path: Path, file_existence_policy: FileExistencePolicy) -> bool:
    """Enforces the file existence policy. Function returns True, if processing should be stopped. Otherwise False.

    Args:
        file_path (Path): File path to the file to check.
        file_existence_policy (FileExistencePolicy): The file existence policy.

    Raises:
        ValueError: Raised if the file existence policy is unknown or the policy requires to raise a ValueError.

    Returns:
        bool: True if processing should be stopped, otherwise False.
    """
    if file_existence_policy == FileExistencePolicy.SKIP:
        get_logger(name="main").warning(f"File already exists at {str(file_path)}. Skipping ...")
        return True
    elif file_existence_policy == FileExistencePolicy.OVERRIDE:
        get_logger(name="main").warning(f"File already exists at {str(file_path)}. Overriding it.")
        os.remove(file_path)
        return False
    elif file_existence_policy == FileExistencePolicy.ERROR:
        raise ValueError("File already exists. Delete it or specify different output folder.")
    else:
        raise ValueError(f"Unknown file existence policy: {file_existence_policy}")


def create_raw_data_index(
    src_path: Path, index_path: Path, file_existence_policy: FileExistencePolicy = FileExistencePolicy.ERROR
):
    """Creates the index file for the content of a large jsonl-file. The index file
    contains the byte-offsets and lengths of each line in the jsonl-file.
    Background is the ability to further process the respective file without loading it,
    while splitting its content line-based. This step is necessary in advance of further processing like tokenization.
    It is only necessary once for a jsonl-file and allows therefore different tokenizations without re-indexing.

    Args:
        src_path (Path): The path to the jsonl-file.
        index_path (Path): The path to the index file, that will be created.
        file_existence_policy (FileExistencePolicy): Policy to apply when the index file already exists.
            Defaults to FileExistencePolicy.ERROR.

    Raises:
        ValueError: If the index file already exists.
    """
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        stop_process = enforce_file_existence_policy(index_path, file_existence_policy)
        if stop_process:
            return

    get_logger(name="main").info(
        f"Reading raw data from {str(src_path)} and" f" writing index to {str(index_path)} ..."
    )
    os.makedirs(index_path.parent, exist_ok=True)

    generator = IndexGenerator(src_path)
    generator.create_index(index_path)


def generate_text(config_file_path: FilePath):
    """Inference function to generate text with a given model.

    Args:
        config_file_path (FilePath): Path to the YAML config file.
    """
    inference.generate_text(config_file_path)


def convert_pytorch_to_hf_checkpoint(
    config_file_path: Path, output_hf_checkpoint_dir: Path, prediction_key: str
) -> HFModelAdapter:
    """Converts a PyTorch checkpoint to a Hugging Face checkpoint.

    Args:
        config_file_path (Path): Path to the config that generated the pytorch checkpoint.
        output_hf_checkpoint_dir (Path): Path to the output directory for the converted HF checkpoint.
        prediction_key (str): The key in the models output where one can find the predictions of interest.

    Returns:
        HFModelAdapter: The Hugging Face model adapter.
    """
    cp = CheckpointConversion(config_file_path, output_hf_checkpoint_dir)
    hf_model = cp.convert_pytorch_to_hf_checkpoint(prediction_key=prediction_key)
    print(f"Model was successfully converted and saved to {output_hf_checkpoint_dir}")
    return hf_model


def shuffle_tokenized_data(
    input_data_path: Path,
    output_data_path: Path,
    batch_size: int,
    file_existence_policy: FileExistencePolicy,
    seed: Optional[int] = None,
):
    """Shuffles a tokenized file (.pbin) and stores it on disc.

    Args:
        input_data_path (Path): File path to the tokenized data (.pbin).
        output_data_path (Path): File path to write the shuffled tokenized data.
        batch_size (int): Number of documents to process per batch.
        file_existence_policy (FileExistencePolicy): Policy to apply when the output file already exists.
        seed (Optional[int]): The seed to use for shuffling.
    """
    if output_data_path.exists():
        stop_process = enforce_file_existence_policy(output_data_path, file_existence_policy)
        if not stop_process:
            return

    DataShuffler.shuffle_tokenized_data(
        input_data_path=input_data_path, output_data_path=output_data_path, batch_size=batch_size, seed=seed
    )


def shuffle_jsonl_data(
    input_data_path: Path,
    output_data_path: Path,
    file_existence_policy: FileExistencePolicy,
    seed: Optional[int] = None,
):
    """Shuffles a JSONL file (.jsonl) and stores it on disc.

    Args:
        input_data_path (Path): File path to the jsonl data (.jsonl).
        output_data_path (Path): File path to write the shuffled jsonl data.
        file_existence_policy (FileExistencePolicy): Policy to apply when the output file already exists.
        seed (Optional[int]): The seed to use for shuffling.
    """
    if output_data_path.exists():
        stop_process = enforce_file_existence_policy(output_data_path, file_existence_policy)
        if not stop_process:
            return

    DataShuffler.shuffle_jsonl_data(input_data_path=input_data_path, output_data_path=output_data_path, seed=seed)


def create_shuffled_dataset_chunk(
    file_path_list: list[Path],
    output_chunk_file_path: Path,
    chunk_id: int,
    num_chunks: int,
    file_existence_policy: FileExistencePolicy,
    global_seed: Optional[int] = None,
):
    """Creates a shuffled dataset chunk.
    Given a dataset consisting of multiple tokenized pbin files, this function
    creates a shuffled dataset chunk for a given chunk id.
    From each tokenized pbin file, the respective chunk is extracted, shuffled
    and written to a new pbin file.

    Args:
        file_path_list (list[Path]): List of paths to the tokenized input pbin files.
        output_chunk_file_path (Path): Path to the output chunk which will be stored in pbin format.
        chunk_id (int): The id of the chunk to create.
        num_chunks (int): The total number of chunks to create.
        file_existence_policy (FileExistencePolicy): Policy to apply when the output chunk file already exists.
        global_seed (Optional[int]): The global seed to use for shuffling.

    Raises:
        ValueError: If the chunk has no samples.
    """
    if output_chunk_file_path.exists():
        stop_process = enforce_file_existence_policy(output_chunk_file_path, file_existence_policy)
        if stop_process:
            return

    samples = []
    token_size_in_bytes = None
    for file_path in tqdm.tqdm(file_path_list, desc=f"Loading file chunks of {chunk_id=}"):
        dataset = PackedMemMapDatasetBase(raw_data_path=file_path, sample_key="text", load_index=True)
        if token_size_in_bytes is None:
            token_size_in_bytes = dataset.token_size_in_bytes
        elif token_size_in_bytes != dataset.token_size_in_bytes:
            raise ValueError("All datasets must have the same token size in bytes.")

        file_samples: list[np.ndarray] = Chunking.get_file_chunk(
            dataset=dataset, num_chunks=num_chunks, chunk_id=chunk_id
        )
        samples.extend(file_samples)

    if len(samples) == 0:
        raise ValueError(
            f"Chunk {chunk_id} has no samples. Please decrease the number of chunks to less than {chunk_id}."
        )

    # samples are shuffled in place
    get_logger(name="main").info(f"Shuffling chunk {chunk_id} ...")
    seed = calculate_hashed_seed(input_data=[str(global_seed), str(chunk_id)]) if global_seed is not None else None
    Chunking.shuffle_file_chunks_in_place(file_chunks=samples, seed=seed)

    get_logger(name="main").info(f"Writing chunk {chunk_id} to {str(output_chunk_file_path)} ...")
    TokenizedFileWriter.write_tokenized_dataset(
        tokenized_dataset=samples,
        tokenized_dataset_file_path=output_chunk_file_path,
        token_size_in_bytes=token_size_in_bytes,
    )
    get_logger(name="main").info(f"Chunk {chunk_id} was successfully written to {str(output_chunk_file_path)}.")


def create_shuffled_jsonl_dataset_chunk(
    file_path_list: list[Path],
    output_chunk_file_path: Path,
    chunk_id: int,
    num_chunks: int,
    file_existence_policy: FileExistencePolicy,
    global_seed: Optional[int] = None,
):
    """Creates a shuffled jsonl dataset chunk.
    Given a dataset consisting of multiple jsonl files, this function
    creates a shuffled dataset chunk for a given chunk id.
    From each jsonl file, the respective chunk is extracted, shuffled
    and written to a new jsonl file.

    Args:
        file_path_list (list[Path]): List of paths to the input jsonl files.
        output_chunk_file_path (Path): Path to the output chunk which will be stored in jsonl format.
        chunk_id (int): The id of the chunk to create.
        num_chunks (int): The total number of chunks to create.
        file_existence_policy (FileExistencePolicy): Policy to apply when the output chunk file already exists.
        global_seed (Optional[int]): The global seed to use for shuffling.

    Raises:
        ValueError: If the chunk has no samples.
    """
    if output_chunk_file_path.exists():
        stop_process = enforce_file_existence_policy(output_chunk_file_path, file_existence_policy)
        if stop_process:
            return

    samples = []
    for file_path in tqdm.tqdm(file_path_list, desc=f"Loading file chunks of {chunk_id=}"):
        with open(file_path, "rb") as f:
            dataset = f.readlines()

        file_samples: list[Any] = Chunking.get_file_chunk(dataset=dataset, num_chunks=num_chunks, chunk_id=chunk_id)
        samples.extend(file_samples)

    if len(samples) == 0:
        raise ValueError(
            f"Chunk {chunk_id} has no samples. Please decrease the number of chunks to less than {chunk_id}."
        )

    # samples are shuffled in place
    get_logger(name="main").info(f"Shuffling chunk {chunk_id} ...")
    seed = calculate_hashed_seed(input_data=[str(global_seed), str(chunk_id)]) if global_seed is not None else None
    Chunking.shuffle_file_chunks_in_place(file_chunks=samples, seed=seed)

    get_logger(name="main").info(f"Writing chunk {chunk_id} to {str(output_chunk_file_path)} ...")
    with open(output_chunk_file_path, "wb") as f:
        for sample in samples:
            f.write(sample)
    get_logger(name="main").info(f"Chunk {chunk_id} was successfully written to {str(output_chunk_file_path)}.")


def pack_encoded_data(
    config_dict: dict,
    file_existence_policy: FileExistencePolicy,
):
    """Packs and encodes an indexed, large jsonl-file.
    (see also `create_index` for more information)
    Returns .pbin-file, which can be inserted into a training process directly
    and does not require its original jsonl-file or the respective index file anymore.

    Args:
        config_dict (dict): Dictionary containing the configuration for the packed data generation.
        file_existence_policy (FileExistencePolicy): Policy to apply when the output file already exists.
    """

    # TODO: if we want to use alternative entrypoints together with the ResolverRegistry,
    #  we can currently not rely on the existing class resolver.
    #  This is based on its connection to the overall `AppConfig`.
    #  One would requires an object of it to instantiate the ResolverRegistry.
    #  This could get resolved by implementing on own ResolverRegistry for each entrypoint or adapting the existing
    #  ResolverRegistry to work dynamically with any type-hinted config object from config.py.
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components: PackedDatasetComponentsInstantiationModel = component_factory.build_components(
        config_dict=config_dict, components_model_type=PackedDatasetComponentsInstantiationModel
    )

    if components.settings.dst_path.exists():
        stop_process = enforce_file_existence_policy(components.settings.dst_path, file_existence_policy)
        if stop_process:
            return

    generator = PackedDataGenerator(
        components.settings.src_path,
        index_path=components.settings.index_path,
        tokenizer=components.tokenizer,
        eod_token=components.settings.eod_token,
        jq_pattern=components.settings.jq_pattern,
        number_of_processes=components.settings.num_cpus,
        processing_batch_size=components.settings.processing_batch_size,
        raw_samples_queue_size=components.settings.raw_samples_queue_size,
        processed_samples_queue_size=components.settings.processed_samples_queue_size,
    )
    generator.run(components.settings.dst_path)


def merge_packed_data_files(src_paths: list[Path], target_path: Path):
    """Utility function for merging different pbin-files into one.
    This is especially useful, if different datasets were at different points in time or if one encoding takes so long,
    that the overall process was done in chunks.
    It is important that the same tokenizer got used for all chunks.

    Specify an arbitrary amount of pbin-files and/or directory containing such as input.

    Args:
        src_paths (list[Path]): List of paths to the pbin-files or directories containing such.
        target_path (Path): The path to the merged pbin-file, that will be created.
    """
    input_files = []
    for p in src_paths:
        p: Path
        if p.is_dir():
            input_files.extend(p.glob("**/*.pbin"))
        else:
            input_files.append(p)
    embedded_datasets = list(map(EmbeddedStreamData, input_files))
    join_embedded_stream_data(embedded_datasets, target_path)
