#!/usr/bin/env python

import json
import os
import socket
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Optional

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
from modalities.dataloader.create_instruction_tuning_data import create_instruction_tuning_data
from modalities.main import Main
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.running_env.cuda_env import CudaEnv
from modalities.util import print_rank_0
from modalities.utils.benchmarking.benchmarking_utils import SweepSets, get_updated_sweep_status
from modalities.utils.benchmarking.sweep_utils import SweepGenerator
from modalities.utils.communication_test import run_communication_test


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
@click.option(
    "--test_comm",
    is_flag=True,
    default=False,
    help="If set, run a communication test before training.",
)
@click.option(
    "--experiment_id",
    type=str,
    default=None,
    help="Optional experiment ID to use for this run. If not provided, it will be derived from the config file path.",
)
@click.option(
    "--error_log_folder",
    type=click_pathlib.Path(),
    default=None,
    help="Optional path to a folder where error logs will be written.",
)
def CMD_entry_point_run_modalities(
    config_file_path: Path,
    test_comm: bool = False,
    experiment_id: Optional[str] = None,
    error_log_folder: Optional[Path] = None,
):
    """Entrypoint to run the model training.

    Args:
        config_file_path (Path): Path to the YAML training config file.
        test_comm (bool): If set, run a communication test before training.
        experiment_id (Optional[str]): Optional experiment ID to use for this run.
            If not provided it will be generated. Default is None.
        error_log_folder (Optional[Path]): Optional path to a folder where error logs will be written.
    """

    def _format_exception_as_json(e: Exception, environment: dict[str, Any]) -> str:
        # Format an exception into a structured JSON string with error message, type, and stack trace.
        error = {
            "error": str(e),
            "type": type(e).__name__,
            "stacktrace": traceback.format_exception(type(e), e, e.__traceback__),
        }

        return json.dumps({"environment": environment, "error": error}, indent=2)

    try:
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            if test_comm:
                print_rank_0("Running communication test...")
                run_communication_test()
                print_rank_0("Communication test succeeded.")

            main_obj = Main(config_file_path, experiment_id=experiment_id)
            components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
            main_obj.run(components)
    except Exception as e:
        if error_log_folder is not None:
            environment = {
                "rank": int(os.environ["RANK"] if "RANK" in os.environ else -1),
                "local_rank": int(os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else -1),
                "world_size": int(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else -1),
                "hostname": socket.gethostname(),
            }
            error_log_folder = (
                error_log_folder.parent
                / f"{error_log_folder.stem}_{environment['hostname']}_{environment['local_rank']}.log"
            )
            error_log_folder.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log_folder, "w", encoding="utf-8") as f:
                f.write(_format_exception_as_json(e, environment))

        raise RuntimeError(f"An error occurred while running the training: {e}. ") from e


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


@data.command(name="prepare_instruction_tuning_data")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to a file with the YAML config file.",
)
def entry_point_data_prepare_instruction_tuning_data(config_file_path: Path):
    """
    Utility for preparing instruction-tuning data by converting, train-val-splitting, index- and pbin-file-creation.
    """
    create_instruction_tuning_data(config_file_path=config_file_path)


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


@main.group(name="benchmark")
def benchmark():
    """
    Collection of utilities to prepare and run benchmarks.
    """
    pass


@benchmark.command(name="prepare_sweep_configs")
@click.option(
    "--sweep_config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the sweep configuration YAML file.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory to save the generated sweep configurations.",
)
@click.option(
    "--world_sizes",
    type=str,
    default="2",
    help="Comma-separated list of world sizes (must not have spaces), e.g. --world_sizes '2,4,8'",
)
def prepare_sweep_configs(sweep_config_path: Path, output_dir: Path, world_sizes: str):
    """
    Utility for preparing sweep configurations.
    """
    try:
        world_sizes_list: list[int] = list(map(int, world_sizes.split(",")))
    except ValueError as e:
        raise ValueError("Invalid world_sizes format. Please provide a comma-separated list of integers.") from e
    SweepGenerator.generate_sweep_configs(sweep_config_path, output_dir, world_sizes_list)


@benchmark.command(name="list_remaining_runs")
@click.option(
    "--exp_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the root directory of the experiment containing config files.",
)
@click.option(
    "--file_list_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file to store paths of configs to run.",
)
@click.option(
    "--expected_steps",
    type=int,
    required=True,
    help="Expected number of steps in evaluation_results.jsonl",
)
@click.option(
    "--skip_exception_types",
    type=str,
    default="",
    help="Exception types to skip when checking for successful runs. "
    "Typically, we would add 'OutOfMemoryError', as rerunning the experiment would result in the same error. "
    " List of exceptions is comma-separated.",
)
def CMD_entry_point_prepare_remaining_runs(
    exp_root: Path,
    file_list_path: Path,
    expected_steps: int,
    skip_exception_types: str = "",
):
    """
    Prepare a file list of remaining runs from a grid search experiment directory.
    """
    skip_exception_types_list = skip_exception_types.split(",") if skip_exception_types != "" else []
    file_list_dict = get_updated_sweep_status(
        exp_root=exp_root,
        expected_steps=expected_steps,
        skip_exception_types=skip_exception_types_list,
    )
    with file_list_path.open("w", encoding="utf-8") as f:
        for cfg in file_list_dict[SweepSets.UPDATED_CONFIGS.value]:
            f.write(f"{cfg}\n")


if __name__ == "__main__":
    main()
