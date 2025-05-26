#!/usr/bin/env python

import json
from functools import partial
from pathlib import Path
from typing import Optional

import click
import click_pathlib
from omegaconf import DictConfig
from pydantic import FilePath

from modalities.api import (
    FileExistencePolicy,
    convert_pytorch_to_hf_checkpoint,
    create_raw_data_index,
    create_shuffled_dataset_chunk,
    create_shuffled_jsonl_dataset_chunk,
    generate_text,
    merge_packed_data_files,
    pack_encoded_data,
    shuffle_jsonl_data,
    shuffle_tokenized_data,
)
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.main import Main
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.running_env.cuda_env import CudaEnv
from modalities.utils.profilers.modalities_profiler import ModalitiesProfiler


@click.group()
def main() -> None:
    pass


@main.command(name="run")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the YAML training config file.",
)
def CMD_entry_point_run_modalities(config_file_path: Path):
    """Entrypoint to run the model training.

    Args:
        config_file_path (Path): Path to the YAML training config file.
    """
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main_obj = Main(config_file_path)
        components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main_obj.run(components)


@main.command(name="warmstart")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the YAML warmstart config file.",
)
@click.option(
    "--last_checkpoint_info_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the file containing the model and optimizer checkpoint paths from the last successful checkpoint.",
)
def CMD_entry_point_warmstart_modalities(config_file_path: Path, last_checkpoint_info_file_path: Path):
    """Entrypoint to run the model warmstart.

    Args:
        config_file_path (Path): Path to the YAML warmstart config file.
        last_checkpoint_info_file_path (Path): Path to the file containing the model and
            optimizer checkpoint paths from the last successful checkpoint.
    """

    def get_last_checkpoint_resolver_fun(var_name: str, last_checkpoint_info_file_path: Path) -> dict[str, str]:
        if var_name != "checkpoint_paths":
            raise ValueError(f"Unknown variable name {var_name}. Should be 'checkpoint_paths'.")
        with open(last_checkpoint_info_file_path, "r") as f:
            last_checkpoint_info = json.load(f)
        return DictConfig(last_checkpoint_info)

    resolver_funs = {
        "warmstart_env": partial(
            get_last_checkpoint_resolver_fun, last_checkpoint_info_file_path=last_checkpoint_info_file_path
        )
    }
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main_obj = Main(config_file_path, additional_resolver_funs=resolver_funs)
        components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main_obj.run(components)


@main.command(name="generate_text")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to a file with the YAML config file.",
)
def CMD_entry_point_generate_text(config_file_path: FilePath):
    """Inference entrypoint to generate text with a given model.

    Args:
        config_file_path (FilePath): Path to the YAML config file.
    """
    generate_text(config_file_path)


@main.command(name="convert_pytorch_to_hf_checkpoint")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to config of model checkpoint.",
)
@click.option(
    "--output_hf_checkpoint_dir",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Converted HF checkpoint will be written to this directory.",
)
@click.option(
    "--prediction_key",
    type=str,
    required=True,
    help="The key in the models output, where one can find the logits.",
)
def CMD_entry_point_convert_pytorch_to_hf_checkpoint(
    config_file_path: Path, output_hf_checkpoint_dir: Path, prediction_key: str
) -> HFModelAdapter:
    """Entrypoint to convert a PyTorch checkpoint to a Hugging Face checkpoint.

    Args:
        config_file_path (Path): Path to the config that generated the pytorch checkpoint.
        output_hf_checkpoint_dir (Path): Path to the output directory for the converted HF checkpoint.
        prediction_key (str): The key in the models output where one can find the predictions of interest.

    Returns:
        HFModelAdapter: The Hugging Face model adapter.
    """
    convert_pytorch_to_hf_checkpoint(
        config_file_path=config_file_path,
        output_hf_checkpoint_dir=output_hf_checkpoint_dir,
        prediction_key=prediction_key,
    )


@main.group(name="data")
def data():
    """
    Collection of utilities to preprocess, analyse and modify training data.
    """
    pass


@data.command(name="create_raw_index")
@click.argument("src_path", type=Path)
@click.option(
    "--index_path",
    type=Path,
    default=None,
    help="output path for index. will use parent directory of src_path if none.",
)
@click.option(
    "--file_existence_policy",
    type=click.Choice([policy.value for policy in FileExistencePolicy]),
    default=FileExistencePolicy.ERROR.value,
    help="Policy for handling existing files.",
)
def CMD_entry_point_data_create_raw_index(src_path: Path, index_path: Path, file_existence_policy: FileExistencePolicy):
    """Utility CMD for indexing the content of a large jsonl-file.
    Background is the ability to further process the respective file without loading it,
    while splitting its content line-based. This step is necessary in advance of further processing like tokenization.
    It is only necessary once for a jsonl-file and allows therefore different tokenizations without re-indexing.

    Args:
        src_path (Path): The path to the jsonl-file.
        index_path (Path): The path to the index file, that will be created.
        file_existence_policy (FileExistencePolicy): Policy for handling existing files.

    Raises:
        ValueError: If the index file already exists.
    """
    file_existence_policy = FileExistencePolicy(file_existence_policy)
    create_raw_data_index(src_path=src_path, index_path=index_path, file_existence_policy=file_existence_policy)


@data.command(name="pack_encoded_data")
@click.argument("config_path", type=FilePath)
@click.option(
    "--file_existence_policy",
    type=click.Choice([policy.value for policy in FileExistencePolicy]),
    default=FileExistencePolicy.ERROR.value,
    help="Policy for handling existing files.",
)
def CMD_entry_point_pack_encoded_data(config_path: FilePath, file_existence_policy: FileExistencePolicy):
    """Utility to encode an indexed, large jsonl-file.
    (see also `create_index` for more information)
    Returns .pbin-file, which can be inserted into a training process directly
    and does not require its original jsonl-file or the respective index file anymore.

    Args:
        config_path (FilePath): Path to the config file describing the tokenization setup.
        file_existence_policy (FileExistencePolicy): Policy for handling existing files.
    """
    file_existence_policy = FileExistencePolicy(file_existence_policy)
    config_dict = load_app_config_dict(config_path)

    pack_encoded_data(config_dict=config_dict, file_existence_policy=file_existence_policy)


@data.command(name="create_shuffled_dataset_chunk")
@click.option(
    "--input_file_list_path",
    type=Path,
    required=True,
    help="Path to the file containing the list of files to be chunked.",
)
@click.option(
    "--input_data_root_path",
    type=Path,
    required=True,
    help="Directory path to the root of the input data.",
)
@click.option(
    "--output_chunk_file_path",
    type=Path,
    required=True,
    help="Path where the chunked dataset will be saved.",
)
@click.option(
    "--chunk_id",
    type=int,
    required=True,
    help="The id of the chunk to be created.",
)
@click.option(
    "--num_chunks",
    type=int,
    required=True,
    help="The number of chunks to create.",
)
@click.option(
    "--file_existence_policy",
    type=click.Choice([policy.value for policy in FileExistencePolicy]),
    default=FileExistencePolicy.ERROR.value,
    help="Policy for handling existing files.",
)
@click.option(
    "--global_seed",
    type=int,
    default=None,
    help="The global seed to use for shuffling.",
)
def CMD_create_shuffled_dataset_chunk(
    input_file_list_path: Path,
    input_data_root_path: Path,
    output_chunk_file_path: Path,
    chunk_id: int,
    num_chunks: int,
    file_existence_policy: FileExistencePolicy,
    global_seed: Optional[int],
):
    """Utility to create a dataset chunk from a list of shuffled and tokenized pbin files.

    Args:
        input_file_list_path (Path): Path to file that contains relative paths of
            pbin files to be chunked (one per line).
        input_data_root_path (Path): Path to the root directory that contains the files to be chunked.
        output_chunk_file_path (Path): File path to the chunked dataset.
        chunk_id (int): The id of the chunk to be created.
        num_chunks (int): Number of chunks in total.
        file_existence_policy (FileExistencePolicy): Policy for handling existing files.
        global_seed (Optional[int]): The global seed to use for shuffling.
    """
    file_existence_policy = FileExistencePolicy(file_existence_policy)

    with open(input_file_list_path, "r", encoding="utf-8") as f:
        file_path_list = f.readlines()
    file_path_list = [
        input_data_root_path / Path(file_path.strip()).with_suffix(".pbin") for file_path in file_path_list
    ]

    create_shuffled_dataset_chunk(
        file_path_list=file_path_list,
        output_chunk_file_path=output_chunk_file_path,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        file_existence_policy=file_existence_policy,
        global_seed=global_seed,
    )


@data.command(name="create_shuffled_jsonl_chunk")
@click.option(
    "--input_file_list_path",
    type=Path,
    required=True,
    help="Path to the file containing the list of jsonl files to be chunked.",
)
@click.option(
    "--input_data_root_path",
    type=Path,
    required=True,
    help="Directory path to the root of the input data.",
)
@click.option(
    "--output_chunk_file_path",
    type=Path,
    required=True,
    help="Path where the chunked jsonl dataset will be saved.",
)
@click.option(
    "--chunk_id",
    type=int,
    required=True,
    help="The id of the chunk to be created.",
)
@click.option(
    "--num_chunks",
    type=int,
    required=True,
    help="The number of chunks to create.",
)
@click.option(
    "--file_existence_policy",
    type=click.Choice([policy.value for policy in FileExistencePolicy]),
    default=FileExistencePolicy.ERROR.value,
    help="Policy for handling existing files.",
)
@click.option(
    "--global_seed",
    type=int,
    default=None,
    help="The global seed to use for shuffling.",
)
def CMD_create_shuffled_jsonl_dataset_chunk(
    input_file_list_path: Path,
    input_data_root_path: Path,
    output_chunk_file_path: Path,
    chunk_id: int,
    num_chunks: int,
    file_existence_policy: FileExistencePolicy,
    global_seed: Optional[int],
):
    """Utility to create a shuffled jsonl dataset chunk from a list of jsonl files.

    Args:
        input_file_list_path (Path): Path to file that contains relative paths of
            jsonl files to be chunked and shuffled (one per line).
        input_data_root_path (Path): Path to the root directory that contains the jsonl files to be chunked.
        output_chunk_file_path (Path): File path to the chunked jsonl dataset.
        chunk_id (int): The id of the chunk to be created.
        num_chunks (int): Number of chunks in total.
        file_existence_policy (FileExistencePolicy): Policy for handling existing files.
        global_seed (Optional[int]): The global seed to use for shuffling.
    """
    file_existence_policy = FileExistencePolicy(file_existence_policy)

    with open(input_file_list_path, "r", encoding="utf-8") as f:
        file_path_list = f.readlines()
    file_path_list = [
        input_data_root_path / Path(file_path.strip()).with_suffix(".jsonl") for file_path in file_path_list
    ]

    create_shuffled_jsonl_dataset_chunk(
        file_path_list=file_path_list,
        output_chunk_file_path=output_chunk_file_path,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        file_existence_policy=file_existence_policy,
        global_seed=global_seed,
    )


@data.command(name="merge_packed_data")
@click.argument("src_paths", type=click.types.Path(exists=True, path_type=Path), nargs=-1, required=True)
@click.argument("target_path", type=click.types.Path(file_okay=False, dir_okay=False, path_type=Path))
def CMD_entry_point_merge_packed_data(src_paths: list[Path], target_path: Path):
    """Utility for merging different pbin-files into one.
    This is especially useful, if different datasets were at different points in time or if one encoding takes so long,
    that the overall process was done in chunks.
    It is important that the same tokenizer got used for all chunks.

    Specify an arbitrary amount of pbin-files and/or directory containing such as input.

    Args:
        src_paths (list[Path]): List of paths to the pbin-files or directories containing such.
        target_path (Path): The path to the merged pbin-file, that will be created.
    """
    merge_packed_data_files(src_paths=src_paths, target_path=target_path)


@data.command(name="shuffle_tokenized_data")
@click.option(
    "--input_data_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to a tokenized file (.pbin).",
)
@click.option(
    "--output_data_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to write the shuffled tokenized data (.pbin).",
)
@click.option(
    "--batch_size", type=int, default=100, show_default=True, help="Number of documents to process per batch."
)
@click.option(
    "--file_existence_policy",
    type=click.Choice([policy.value for policy in FileExistencePolicy]),
    default=FileExistencePolicy.ERROR.value,
    help="Policy for handling existing files.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="The seed for shuffling the data.",
)
def CMD_shuffle_tokenized_data(
    input_data_path: Path, output_data_path: Path, batch_size: int, file_existence_policy, seed: int
) -> None:
    """Entrypoint for shuffling tokenized data.

    Args:
        input_data_path (Path): The path to the input tokenized data (.pbin).
        output_data_path (Path): File path to write the shuffled tokenized data (.pbin).
        batch_size (int): The size of the batches to shuffle.
        file_existence_policy (FileExistencePolicy): Policy for handling existing files.
        seed (int): The seed for shuffling the data.
    Returns:
        None
    """
    file_existence_policy = FileExistencePolicy(file_existence_policy)

    shuffle_tokenized_data(
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        batch_size=batch_size,
        file_existence_policy=file_existence_policy,
        seed=seed,
    )


@data.command(name="shuffle_jsonl_data")
@click.option(
    "--input_data_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to a jsonl file (.jsonl).",
)
@click.option(
    "--output_data_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to write the shuffled jsonl data (.jsonl).",
)
@click.option(
    "--file_existence_policy",
    type=click.Choice([policy.value for policy in FileExistencePolicy]),
    default=FileExistencePolicy.ERROR.value,
    help="Policy for handling existing files.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="The seed for shuffling the data.",
)
def CMD_shuffle_jsonl_data(
    input_data_path: Path, output_data_path: Path, file_existence_policy, seed: Optional[int]
) -> None:
    """Entrypoint for shuffling jsonl data.

    Args:
        input_data_path (Path): The path to the input jsonl data (.jsonl).
        output_data_path (Path): File path to write the shuffled jsonl data (.jsonl).
        file_existence_policy (FileExistencePolicy): Policy for handling existing files.
        seed (Optional[int]): The seed for shuffling the data. Default is None.
    Returns:
        None
    """
    file_existence_policy = FileExistencePolicy(file_existence_policy)

    shuffle_jsonl_data(
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        file_existence_policy=file_existence_policy,
        seed=seed,
    )


@main.group(name="profile")
def profile():
    """
    Collection of utilities to profile modalities.
    """
    pass


@profile.command(name="train_step")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the YAML training config file.",
)
@click.option(
    "--experiment_folder_path",
    type=click_pathlib.Path(file_okay=False),
    required=True,
    help="Path to the experiment output directory.",
)
@click.option(
    "--num_warmup_steps",
    type=int,
    default=1,
    show_default=True,
    help="Number of warmup steps to skip in profiling.",
)
@click.option(
    "--num_measurement_steps",
    type=int,
    default=3,
    show_default=True,
    help="Number of steps to measure during profiling.",
)
def CMD_entry_point_run_train_step_profiler(
    config_file_path: Path,
    experiment_folder_path: Path,
    num_warmup_steps: int,
    num_measurement_steps: int,
):
    """Run train step profiler and write result to JSON if RANK=0."""
    ModalitiesProfiler.get_train_step_statistics(
        config_file_path=config_file_path,
        experiment_folder_path=experiment_folder_path,
        num_warmup_steps=num_warmup_steps,
        num_measurement_steps=num_measurement_steps,
    )


if __name__ == "__main__":
    main()
