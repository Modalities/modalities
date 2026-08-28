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
from modalities.dataloader.preprocessing.quality import export as quality_export
from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline
from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry
from modalities.dataloader.preprocessing.quality.verify import format_verify_report
from modalities.main import Main
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.running_env.cuda_env import CudaEnv
from modalities.util import print_rank_0
from modalities.utils.benchmarking.benchmarking_utils import SweepSets, get_updated_sweep_status
from modalities.utils.benchmarking.sweep_utils import SweepGenerator
from modalities.utils.communication_test import run_communication_test
from modalities.utils.logger_utils import get_logger
from modalities.utils.profilers.modalities_profiler import ModalitiesProfilerStarter

logger = get_logger("__main__")


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
    "--experiments_root_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the root directory where experiment folders will be created.",
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
@click.option(
    "--test_comm",
    is_flag=True,
    default=False,
    help="If set, run a communication test before training.",
)
def CMD_entry_point_run_modalities(
    config_file_path: Path,
    experiments_root_path: Path,
    experiment_id: Optional[str] = None,
    error_log_folder: Optional[Path] = None,
    test_comm: bool = False,
):
    """Entrypoint to run the model training.

    Args:
        config_file_path (Path): Path to the YAML training config file.
        experiments_root_path (Path): Path to the root directory where experiment folders will be created.
        experiment_id (Optional[str]): Optional experiment ID to use for this run.
            If not provided it will be generated. Default is None.
        error_log_folder (Optional[Path]): Optional path to a folder where error logs will be written.
        test_comm (bool): If set, run a communication test before training.
    """

    try:
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            if test_comm:
                print_rank_0("Running communication test...")
                run_communication_test()
                print_rank_0("Communication test succeeded.")

            main_obj = Main(config_file_path, experiments_root_path=experiments_root_path, experiment_id=experiment_id)
            components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
            main_obj.run(components)
    except Exception as e:
        _exception_handling(e, error_log_folder)


@main.command(name="warmstart")
@click.option(
    "--experiments_root_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the root directory where experiment folders will be created.",
)
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
@click.option(
    "--error_log_folder",
    type=click_pathlib.Path(),
    default=None,
    help="Optional path to a folder where error logs will be written.",
)
def CMD_entry_point_warmstart_modalities(
    experiments_root_path: Path,
    config_file_path: Path,
    last_checkpoint_info_file_path: Path,
    error_log_folder: Optional[Path] = None,
):
    """Entrypoint to run the model warmstart.

    Args:
        experiments_root_path (Path): Path to the root directory where experiment folders will be created.
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
    try:
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            main_obj = Main(
                config_file_path, experiments_root_path=experiments_root_path, additional_resolver_funs=resolver_funs
            )
            components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
            main_obj.run(components)
    except Exception as e:
        _exception_handling(e, error_log_folder)


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
    "--world_size",
    type=int,
    required=False,
    default=None,
    help="Number of ranks (world size) to filter the configs for.",
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
    "--create_new_folders_if_partially_done",
    is_flag=True,
    default=False,
    help="Create new experiment folders for remaining configs if some runs already exist.",
)
@click.option(
    "--skip_exception_types",
    type=str,
    default="",
    help="Exception types to skip when checking for successful runs. "
    "Typically, we would add 'OutOfMemoryError', as rerunning the experiment would result in the same error. "
    " List of exceptions is comma-separated.",
)
def CMD_entry_point_list_remaining_runs(
    exp_root: Path,
    file_list_path: Path,
    expected_steps: int,
    create_new_folders_if_partially_done: bool,
    world_size: int | None = None,
    skip_exception_types: str = "",
):
    """
    Prepare a file list of remaining runs from a grid search experiment directory.
    """
    skip_exception_types_list = skip_exception_types.split(",") if skip_exception_types != "" else []
    file_list_dict = get_updated_sweep_status(
        exp_root=exp_root,
        world_size=world_size,
        expected_steps=expected_steps,
        skip_exception_types=skip_exception_types_list,
        create_new_folders_if_partially_done=create_new_folders_if_partially_done,
    )
    if SweepSets.UPDATED_CONFIGS.value in file_list_dict:
        with file_list_path.open("w", encoding="utf-8") as f:
            for cfg in file_list_dict[SweepSets.UPDATED_CONFIGS.value]:
                f.write(f"{cfg}\n")


@main.group(name="profile")
def profile():
    """
    Collection of utilities to profile modalities.
    """
    pass


@profile.command(name="distributed")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the YAML training config file.",
)
@click.option(
    "--experiment_root_path",
    type=click_pathlib.Path(file_okay=False),
    required=True,
    help="Path to the experiment output directory.",
)
def CMD_entry_point_run_train_step_profiler(
    config_file_path: Path,
    experiment_root_path: Path,
):
    """Run train step profiler and write result to JSON if RANK=0."""
    ModalitiesProfilerStarter.run_distributed(
        config_file_path=config_file_path,
        experiment_root_path=experiment_root_path,
    )


@main.group(name="quality")
def quality() -> None:
    """
    Quality-based document selection and up/downsampling of a training blend.

    The stages are meant to be run in order: `calibrate` measures how records map to
    token counts, `build-sidecar` records one row per document, `join-annotations`
    attaches external labels, and `build-cube` aggregates the result. After that,
    `preview` costs a selection in seconds and `apply` writes filtered index files that
    `modalities data pack_encoded_data` consumes unchanged.
    """
    pass


@quality.command(name="calibrate")
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory for the blend's intermediates.")
@click.option(
    "--tokenizer_config",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to a packing config whose tokenizer section is used for calibration.",
)
@click.option(
    "--sample_size",
    type=int,
    default=4000,
    show_default=True,
    help="Documents tokenized per dataset to measure the estimator. Sampled in proportion to "
    "document length and grouped into size strata, so the rare very long documents that "
    "dominate a corpus's byte count are represented.",
)
@click.option("--only", multiple=True, help="Restrict to these dataset names (repeatable).")
def CMD_quality_calibrate(
    registry_path: Path, work_dir: Path, tokenizer_config: Path, sample_size: int, only: tuple[str, ...]
) -> None:
    """Measures how each dataset's records relate to the training tokenizer's counts.

    Args:
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory for the blend's intermediates.
        tokenizer_config (Path): Packing config supplying the tokenizer.
        sample_size (int): Documents tokenized per dataset.
        only (tuple[str, ...]): Restrict to these dataset names.
        resume (bool): Skip files whose part already reads as valid parquet.
    """
    from modalities.config.component_factory import ComponentFactory
    from modalities.config.instantiation_models import TokenizerInstantiationModel
    from modalities.registry.components import COMPONENTS
    from modalities.registry.registry import Registry

    config_dict = load_app_config_dict(tokenizer_config)
    factory = ComponentFactory(registry=Registry(COMPONENTS))
    tokenizer = factory.build_components(
        config_dict=config_dict, components_model_type=TokenizerInstantiationModel
    ).tokenizer
    tokenizer_name = str(config_dict["tokenizer"]["config"].get("pretrained_model_name_or_path", "unknown"))

    quality_pipeline.calibrate_blend(
        registry=CorpusRegistry.from_yaml(registry_path),
        work_dir=work_dir,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        sample_size=sample_size,
        only=list(only) or None,
    )


@quality.command(name="build-sidecar")
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory for the blend's intermediates.")
@click.option("--only", multiple=True, help="Restrict to these dataset names (repeatable).")
@click.option(
    "--index_root",
    type=Path,
    default=None,
    help="Where JSONL index files live or should be created. Use this when the source tree is read-only.",
)
@click.option(
    "--shard_id",
    type=int,
    default=0,
    show_default=True,
    help="This task's index. Set from SLURM_ARRAY_TASK_ID to run the build as an array.",
)
@click.option(
    "--num_shards",
    type=int,
    default=1,
    show_default=True,
    help="How many tasks share the work. Files are divided across all selected datasets, "
    "so one array covers the whole blend.",
)
@click.option(
    "--resume/--no_resume",
    default=False,
    help="Skip files whose sidecar part already reads as valid parquet. Parquet writes its "
    "footer last, so a part left by a killed task fails to open and is rebuilt. Off by "
    "default: after changing a native metric the existing parts are stale and must be redone.",
)
def CMD_quality_build_sidecar(
    registry_path: Path,
    work_dir: Path,
    only: tuple[str, ...],
    index_root: Optional[Path],
    shard_id: int,
    num_shards: int,
    resume: bool,
) -> None:
    """Records one row per document: position, estimated tokens, key and native metrics.

    The only stage that reads the raw data, so the one worth running as an array. Each
    task writes its own parquet parts, so tasks never contend.

    Args:
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory for the blend's intermediates.
        only (tuple[str, ...]): Restrict to these dataset names.
        index_root (Optional[Path]): Where JSONL index files live or should be created.
        shard_id (int): This task's index in [0, num_shards).
        num_shards (int): Total number of tasks sharing the work.
    """
    written = quality_pipeline.build_sidecars(
        registry=CorpusRegistry.from_yaml(registry_path),
        work_dir=work_dir,
        only=list(only) or None,
        index_root=index_root,
        shard_id=shard_id,
        num_shards=num_shards,
        resume=resume,
    )
    for name, n_documents in written.items():
        print_rank_0(f"{name}: {n_documents:,} documents")


@quality.command(name="bucket-annotations")
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory for the blend's intermediates.")
@click.option("--only", multiple=True, help="Restrict to the splits these datasets need (repeatable).")
@click.option(
    "--num_buckets",
    type=int,
    default=1024,
    show_default=True,
    help="Partitions per annotation split. Each is loaded whole during the join, so raise this for large splits.",
)
@click.option(
    "--shard_id",
    type=int,
    default=0,
    show_default=True,
    help="This task's index. Set from SLURM_ARRAY_TASK_ID to bucket as an array.",
)
@click.option("--num_shards", type=int, default=1, show_default=True, help="How many tasks bucket each split.")
@click.option("--force", is_flag=True, default=False, help="Re-bucket a split even if its output is complete.")
def CMD_quality_bucket_annotations(
    registry_path: Path,
    work_dir: Path,
    only: tuple[str, ...],
    num_buckets: int,
    shard_id: int,
    num_shards: int,
    force: bool,
) -> None:
    """Partitions the annotation splits by a hash of their key, ready for joining.

    The expensive half of the join, since a split can run to billions of rows. Shardable,
    and splits shared by several datasets are bucketed only once.

    Args:
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory for the blend's intermediates.
        only (tuple[str, ...]): Restrict to the splits these datasets need.
        num_buckets (int): Partitions per split.
        shard_id (int): This task's index in [0, num_shards).
        num_shards (int): How many tasks bucket each split.
        force (bool): Re-bucket even if the output is complete.
    """
    written = quality_pipeline.bucket_blend_annotations(
        registry=CorpusRegistry.from_yaml(registry_path),
        work_dir=work_dir,
        only=list(only) or None,
        n_buckets=num_buckets,
        shard_id=shard_id,
        num_shards=num_shards,
        force=force,
    )
    for split, n_rows in written.items():
        print_rank_0(f"{split}: {n_rows:,} rows bucketed by this task")


@quality.command(name="join-annotations")
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory for the blend's intermediates.")
@click.option("--only", multiple=True, help="Restrict to these dataset names (repeatable).")
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip sidecar parts that already carry labels, to continue an interrupted run. "
    "Omit it after re-bucketing the annotations, or the old labels are kept.",
)
def CMD_quality_join_annotations(registry_path: Path, work_dir: Path, only: tuple[str, ...], resume: bool) -> None:
    """Attaches the bucketed annotations to each dataset's sidecar and reports coverage.

    Run `bucket-annotations` first. Read the reported coverage before trusting a
    selection: on a partly downloaded split most documents may carry no label at all.

    Args:
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory for the blend's intermediates.
        only (tuple[str, ...]): Restrict to these dataset names.
        resume (bool): Skip parts that already carry labels.
    """
    reports = quality_pipeline.join_blend_annotations(
        registry=CorpusRegistry.from_yaml(registry_path),
        work_dir=work_dir,
        only=list(only) or None,
        resume=resume,
    )
    for report in reports:
        print_rank_0(report.summary())


@quality.command(name="verify-sidecar")
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory holding the sidecars.")
@click.option("--only", multiple=True, help="Restrict to these dataset names (repeatable).")
@click.option(
    "--num_parts",
    type=int,
    default=8,
    show_default=True,
    help="Sidecar parts to sample per dataset.",
)
@click.option(
    "--num_rows_per_part",
    type=int,
    default=4,
    show_default=True,
    help="Documents to probe per sampled part. Rows at offset 0 are skipped, since the first "
    "document of any JSONL file parses and so proves nothing.",
)
@click.option(
    "--adopt",
    is_flag=True,
    default=False,
    help="Write a source file manifest for verified sidecars that lack one, so later stages can "
    "detect drift cheaply. Only sidecars that pass verification are stamped.",
)
def CMD_quality_verify_sidecar(
    registry_path: Path,
    work_dir: Path,
    only: tuple[str, ...],
    num_parts: int,
    num_rows_per_part: int,
    adopt: bool,
) -> None:
    """Checks that the sidecars' byte offsets still describe the current source files.

    Run this after any data transfer, and before apply on a blend whose source tree may
    have been touched. A corpus that was re-sharded after its sidecar was built yields a
    blend of wrong byte ranges, and this is the only check that reads the source bytes.

    Args:
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory holding the sidecars.
        only (tuple[str, ...]): Restrict to these dataset names.
        num_parts (int): Sidecar parts to sample per dataset.
        num_rows_per_part (int): Documents to probe per sampled part.
        adopt (bool): Stamp a manifest onto verified sidecars that lack one.
    """
    reports = quality_pipeline.verify_sidecars(
        registry=CorpusRegistry.from_yaml(registry_path),
        work_dir=work_dir,
        only=list(only) or None,
        n_parts=num_parts,
        n_rows_per_part=num_rows_per_part,
        adopt=adopt,
    )
    print_rank_0(format_verify_report(reports))
    if any(not r.ok for r in reports):
        raise SystemExit(1)


@quality.command(name="build-cube")
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory for the blend's intermediates.")
@click.option("--only", multiple=True, help="Restrict to these dataset names (repeatable).")
@click.option(
    "--num_score_bins",
    type=int,
    default=10,
    show_default=True,
    help="Quantile bins per native metric. A threshold on a bin edge stays exact.",
)
@click.option(
    "--label_dimension",
    "label_dimensions",
    multiple=True,
    help="Annotation column to group on, repeatable. Defaults to the seven ordinal fields; "
    "name a field here if a selection thresholds on it, or the preview must scan the sidecar. "
    "Each added field multiplies the cell count by its number of levels.",
)
def CMD_quality_build_cube(
    registry_path: Path,
    work_dir: Path,
    only: tuple[str, ...],
    num_score_bins: int,
    label_dimensions: tuple[str, ...],
) -> None:
    """Aggregates the sidecars so a selection can be costed without reading them again.

    Args:
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory for the blend's intermediates.
        only (tuple[str, ...]): Restrict to these dataset names.
        num_score_bins (int): Quantile bins per native metric.
        label_dimensions (tuple[str, ...]): Annotation columns to group on.
    """
    quality_pipeline.build_cubes(
        registry=CorpusRegistry.from_yaml(registry_path),
        work_dir=work_dir,
        only=list(only) or None,
        n_score_bins=num_score_bins,
        label_dimensions=list(label_dimensions) or None,
    )


@quality.command(name="preview")
@click.option(
    "--selection",
    "selection_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the selection YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory holding the cubes and sidecars.")
@click.option(
    "--exact",
    is_flag=True,
    default=False,
    help="Scan the per-document sidecars instead of the cubes. Slower, but exact for any threshold.",
)
@click.option(
    "--allow_fallback",
    is_flag=True,
    default=False,
    help="Let a dataset whose cube cannot answer a predicate be scanned from its sidecar. "
    "That reads every document, so it costs minutes to hours rather than seconds.",
)
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Attribute each dataset's retention to its individual predicates, and show how they "
    "overlap. Answers which condition binds and which is redundant.",
)
def CMD_quality_preview(
    selection_path: Path, work_dir: Path, exact: bool, allow_fallback: bool, explain: bool
) -> None:
    """Reports how many documents and tokens a selection yields, per dataset and in total.

    Args:
        selection_path (Path): Path to the selection YAML.
        work_dir (Path): Working directory holding the cubes and sidecars.
        exact (bool): Scan the sidecars instead of the cubes.
        allow_fallback (bool): Permit per-dataset sidecar scans where a cube falls short.
        explain (bool): Attribute retention to individual predicates.
    """
    _, report = quality_pipeline.preview_selection(
        selection_path=selection_path,
        work_dir=work_dir,
        force_exact=exact,
        allow_sidecar_fallback=allow_fallback,
        explain=explain,
    )
    print_rank_0(report)


@quality.command(name="apply")
@click.option(
    "--selection",
    "selection_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the selection YAML.",
)
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--work_dir", type=Path, required=True, help="Working directory holding the sidecars.")
@click.option(
    "--output_dir", type=Path, required=True, help="Directory receiving the filtered index files and the mix manifest."
)
@click.option(
    "--allow_overexposure",
    is_flag=True,
    default=False,
    help="Materialise even when the run would repeat data past a declared cap. Off by default: "
    "ratios are per pass, so a run that wraps multiplies every one of them.",
)
def CMD_quality_apply(
    selection_path: Path,
    registry_path: Path,
    work_dir: Path,
    output_dir: Path,
    allow_overexposure: bool,
) -> None:
    """Writes a selection out as filtered index files plus a manifest.

    The source data is not copied or modified. Point `pack_encoded_data` at a written
    index to tokenize only the selected documents.

    Args:
        selection_path (Path): Path to the selection YAML.
        registry_path (Path): Path to the corpus registry YAML.
        work_dir (Path): Working directory holding the sidecars.
        output_dir (Path): Directory receiving the index files and manifest.
    """
    manifest_path = quality_pipeline.apply_selection(
        allow_overexposure=allow_overexposure,
        selection_path=selection_path,
        registry_path=registry_path,
        work_dir=work_dir,
        output_dir=output_dir,
    )
    print_rank_0(f"Manifest written to {manifest_path}")


@quality.command(name="export-jsonl")
@click.option(
    "--manifest",
    "manifest_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the mix_manifest.yaml written by 'apply'.",
)
@click.option(
    "--registry",
    "registry_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the corpus registry YAML.",
)
@click.option("--output_dir", type=Path, required=True, help="Directory receiving one subdirectory per dataset.")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Overrides the seed the selection was applied with, which decides which documents "
    "receive the extra copy of a fractional repeat factor.",
)
@click.option("--only", multiple=True, help="Restrict to these dataset names (repeatable).")
@click.option(
    "--resume/--no_resume",
    default=True,
    help="Leave shards that are already complete alone (default: resume).",
)
@click.option(
    "--finalize/--no_finalize",
    default=True,
    help="Merge the per-dataset records into export_manifest.yaml. Pass --no_finalize in an "
    "array task, then run 'export-jsonl --finalize_only' once the array has finished.",
)
@click.option(
    "--finalize_only",
    is_flag=True,
    help="Write export_manifest.yaml from the per-dataset records already on disk, exporting nothing.",
)
@click.option("--shard_id", type=int, default=0, show_default=True, help="This task's index in [0, num_shards).")
@click.option(
    "--num_shards",
    type=int,
    default=1,
    show_default=True,
    help="Split each dataset's source files across this many tasks. One task per dataset is fine "
    "until one dataset holds far more files than the rest.",
)
def CMD_quality_export_jsonl(
    manifest_path: Path,
    registry_path: Path,
    output_dir: Path,
    seed: Optional[int],
    only: tuple[str, ...],
    resume: bool,
    finalize: bool,
    finalize_only: bool,
    shard_id: int,
    num_shards: int,
) -> None:
    """Writes the selected documents out as JSONL, with the sampling baked into the bytes.

    Up- and downsampling is materialised here: a dataset at 3.0 has each of its documents
    written three times, and one at 0.6 loses two of every five. The training set is the
    concatenation of the resulting files, so their ratios must not be applied again.

    Args:
        manifest_path (Path): Path to the mix manifest.
        registry_path (Path): Path to the corpus registry YAML.
        output_dir (Path): Directory receiving the exported JSONL.
        seed (Optional[int]): Overrides the selection's seed.
        only (tuple[str, ...]): Restrict to these dataset names.
        resume (bool): Leave complete shards alone.
        finalize (bool): Merge the per-dataset records afterwards.
        finalize_only (bool): Only merge the records; export nothing.
        shard_id (int): This task's index.
        num_shards (int): Tasks splitting each dataset's files.
    """
    if finalize_only:
        print_rank_0(f"Export manifest written to {quality_export.finalize_export(output_dir)}")
        return

    exports = quality_pipeline.export_jsonl(
        manifest_path=manifest_path,
        registry_path=registry_path,
        output_dir=output_dir,
        seed=seed,
        only=list(only) or None,
        resume=resume,
        finalize=finalize,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    n_lines = sum(e.n_lines for e in exports)
    n_bytes = sum(e.n_bytes for e in exports)
    skipped = sum(1 for e in exports for s in e.shards if s.skipped)
    print_rank_0(
        f"Exported {len(exports)} dataset(s): {n_lines:,} lines, {n_bytes / 1e12:,.2f} TB"
        + (f", {skipped:,} shard(s) already complete" if skipped else "")
    )


def _format_exception_as_json(e: Exception, environment: dict[str, Any]) -> str:
    # Format an exception into a structured JSON string with error message, type, and stack trace.
    error = {
        "error": str(e),
        "type": type(e).__name__,
        "stacktrace": traceback.format_exception(type(e), e, e.__traceback__),
    }
    return json.dumps({"environment": environment, "error": error}, indent=2)


def _exception_handling(e: Exception, error_log_folder: Path | None):
    if error_log_folder is not None:
        environment = {
            "rank": int(os.environ["RANK"] if "RANK" in os.environ else -1),
            "local_rank": int(os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else -1),
            "world_size": int(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else -1),
            "hostname": socket.gethostname(),
        }
        error_log_folder = error_log_folder / f"error_logs_{environment['hostname']}_{environment['local_rank']}.log"
        error_log_folder.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_folder, "w", encoding="utf-8") as f:
            f.write(_format_exception_as_json(e, environment))

    raise RuntimeError(f"An error occurred while running the training: {e}. ") from e


if __name__ == "__main__":
    main()
