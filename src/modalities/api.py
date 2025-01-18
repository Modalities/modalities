#!/usr/bin/env python

import os
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import FilePath

import modalities.inference.inference as inference
from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.instantiation_models import PackedDatasetComponentsInstantiationModel
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.dataloader.preprocessing.chunking.create_chunks import Chunking
from modalities.dataloader.preprocessing.tokenization.tokenized_file_writer import TokenizedFileWriter
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.utils.logging import get_logger


class FileExistencePolicy(Enum):
    SKIP = "skip"
    ERROR = "error"
    OVERRIDE = "override"


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

    Raises:
        ValueError: If the index file already exists.
    """
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        if file_existence_policy == FileExistencePolicy.SKIP:
            get_logger(name="main").warning(f"Index already exists at {str(index_path)}. Skipping index creation.")
            return
        elif file_existence_policy == FileExistencePolicy.OVERRIDE:
            get_logger(name="main").warning(f"Index already exists at {str(index_path)}. Overriding it.")
            os.remove(index_path)
        elif file_existence_policy == FileExistencePolicy.ERROR:
            raise ValueError("index already exists. delete it or specify different output folder.")
        else:
            raise ValueError(f"Unknown file existence policy: {file_existence_policy}")

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


def create_shuffled_dataset_chunk(
    file_path_list: list[Path], chunk_file_path: Path, chunk_id: int, num_chunks: int, vocab_size: int
):
    """Creates a shuffled dataset chunk.
    Given a dataset consisting of multiple tokenized pbin files, this function
    creates a shuffled dataset chunk for a given chunk id.
    From each tokenized pbin file, the respective chunk is extracted, shuffled
    and written to a new pbin file.
    """
    sample_key = "text"
    samples = []
    for file_path in file_path_list:
        dataset = PackedMemMapDatasetBase(raw_data_path=file_path, sample_key=sample_key, load_index=True)
        file_samples: list[np.ndarray] = Chunking.get_file_chunk(dataset, num_chunks=num_chunks, chunk_id=chunk_id)
        samples.extend(file_samples)

    shuffled_samples: list[np.ndarray] = Chunking.shuffle_file_chunks(samples)
    TokenizedFileWriter.write_tokenized_dataset(
        tokenized_dataset=shuffled_samples, tokenized_dataset_file_path=chunk_file_path, vocab_size=vocab_size
    )


def pack_encoded_data(config_dict: dict):
    """Packs and encodes an indexed, large jsonl-file.
    (see also `create_index` for more information)
    Returns .pbin-file, which can be inserted into a training process directly
    and does not require its original jsonl-file or the respective index file anymore.

    Args:
        config_dict (dict): Dictionary containing the configuration for the packed data generation.
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
