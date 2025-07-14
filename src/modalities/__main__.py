#!/usr/bin/env python

import json
import logging
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Type

import click
import click_pathlib
import yaml
from omegaconf import DictConfig
from pydantic import BaseModel, FilePath

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
from modalities.batch import EvaluationResultBatch
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel, TrainingReportGenerator
from modalities.dataloader.create_instruction_tuning_data import create_instruction_tuning_data
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.trainer import Trainer
from modalities.util import get_experiment_id_of_run, get_total_number_of_trainable_parameters, print_rank_0
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
def CMD_entry_point_run_modalities(config_file_path: Path, test_comm: bool = False):
    """Entrypoint to run the model training.

    Args:
        config_file_path (Path): Path to the YAML training config file.
    """
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        if test_comm:
            print_rank_0("Running communication test...")
            run_communication_test()
            print_rank_0("Communication test succeeded.")

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


class Main:
    """Main class that orchestrates the training process."""

    def __init__(self, config_path: Path, additional_resolver_funs: Optional[dict[str, Callable]] = None) -> None:
        experiment_id = get_experiment_id_of_run(config_path)
        self.config_dict = load_app_config_dict(
            config_file_path=config_path, experiment_id=experiment_id, additional_resolver_funs=additional_resolver_funs
        )
        self.config_path = config_path

        self.registry = Registry(COMPONENTS)
        self.component_factory = ComponentFactory(registry=self.registry)

    def add_custom_component(
        self, component_key: str, variant_key: str, custom_component: Type, custom_config: Type
    ) -> None:
        """Add a custom component to the registry.

        This method comes in especially handy
        when Modalities is used as a library and the user wants to add custom components
        (e.g., custom model or custom loss function) to the registry.

        Args:
            component_key (str): Key of the component to be added to the registry
            variant_key (str): Key of the variant to be added to the registry
            custom_component (Type): The class type of the custom component
            custom_config (Type): The pydantic config type of the custom component
        """
        self.registry.add_entity(
            component_key=component_key,
            variant_key=variant_key,
            component_type=custom_component,
            component_config_type=custom_config,
        )

    def build_components(self, components_model_type: Type[BaseModel]) -> BaseModel:
        """Given a pydantic basemodel, this method builds the components specified in the config file.

        Depending on the use case (e.g., training, inference, etc.), the user can pass different pydantic base models.
        For instance, for tokenization, the basemodel would only have the tokenization-related components specified.

        Args:
            components_model_type (Type[BaseModel]): The pydantic basemodel type that should be
                used to build the components.

        Returns:
            BaseModel: The components built based on the config file.
        """
        components = self.component_factory.build_components(
            config_dict=self.config_dict, components_model_type=components_model_type
        )
        return components

    def run(self, components: TrainingComponentsInstantiationModel):
        """Entrypoint fo running the training process.

        We pass in a TrainingComponentsInstantiationModel,
        which is a pydantic model that contains all the components needed for the training process.

        Args:
            components (TrainingComponentsInstantiationModel): The components needed for the training process.
        """
        # save the config file to the checkpointing path
        if components.settings.cuda_env.global_rank == 0:
            experiment_path = components.settings.paths.checkpoint_saving_path / components.settings.experiment_id
            os.makedirs(experiment_path, exist_ok=True)
            shutil.copy(self.config_path, experiment_path / self.config_path.name)
            resolved_config_path = (experiment_path / self.config_path.name).with_suffix(".yaml.resolved")
            with open(resolved_config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config_dict, f)

        evaluation_result_publisher, progress_publisher = self.get_logging_publishers(
            progress_subscriber=components.progress_subscriber,
            results_subscriber=components.evaluation_subscriber,
            global_rank=components.settings.cuda_env.global_rank,
            local_rank=components.settings.cuda_env.local_rank,
        )

        # Trainer
        global_num_tokens_per_train_step = (
            components.settings.step_profile.local_train_micro_batch_size
            * components.settings.step_profile.sequence_length
            * components.settings.step_profile.gradient_accumulation_steps
            * components.settings.cuda_env.world_size
        )
        trainer = Trainer(
            global_rank=components.settings.cuda_env.global_rank,
            progress_publisher=progress_publisher,
            num_target_steps=components.settings.training_target.num_target_steps,
            num_target_tokens=components.settings.training_target.num_target_tokens,
            num_seen_train_steps=components.settings.training_progress.num_seen_steps,
            global_num_seen_tokens=components.settings.training_progress.global_num_seen_tokens,
            evaluation_result_publisher=evaluation_result_publisher,
            gradient_acc_steps=components.settings.step_profile.gradient_accumulation_steps,
            gradient_clipper=components.gradient_clipper,
            global_num_tokens_per_train_step=global_num_tokens_per_train_step,
            mfu_calculator=components.mfu_calculator,
        )

        # Evaluator
        evaluator = Evaluator(
            progress_publisher=progress_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Gym
        gym = Gym(
            trainer=trainer,
            evaluator=evaluator,
            loss_fun=components.loss_fn,
            num_ranks=components.settings.cuda_env.world_size,
        )
        num_params = get_total_number_of_trainable_parameters(components.app_state.model)
        components.evaluation_subscriber.consume_dict({"No. parameters": num_params})
        logging.info(f"Training model with {num_params} parameters.")

        print_rank_0(f"Model initialized at {datetime.now()}.")

        report = TrainingReportGenerator(
            training_target=components.settings.training_target,
            intervals=components.settings.intervals,
            step_profile=components.settings.step_profile,
            cuda_env=components.settings.cuda_env,
            consistency_enforcement=components.settings.consistency_enforcement,
            train_dataset=components.train_dataset,
            training_progress=components.settings.training_progress,
        ).get_report()

        print_rank_0(report)

        gym.run(
            train_data_loader=components.train_dataloader,
            evaluation_data_loaders=components.eval_dataloaders,
            checkpoint_saving=components.checkpoint_saving,
            app_state=components.app_state,
            checkpointing_interval_in_steps=components.settings.intervals.checkpointing_interval_in_steps,
            evaluation_interval_in_steps=components.settings.intervals.evaluation_interval_in_steps,
            training_log_interval_in_steps=components.settings.intervals.training_log_interval_in_steps,
        )

    def get_logging_publishers(
        self,
        progress_subscriber: MessageSubscriberIF[ProgressUpdate],
        results_subscriber: MessageSubscriberIF[EvaluationResultBatch],
        global_rank: int,
        local_rank: int,
    ) -> tuple[MessagePublisher[EvaluationResultBatch], MessagePublisher[ProgressUpdate]]:
        """Returns the logging publishers for the training.

        These publishers are used to pass the evaluation results and the progress updates to the message broker.
        The message broker is then used to pass the messages to the subscribers, such as WandB.

        Args:
            progress_subscriber (MessageSubscriberIF[ProgressUpdate]): The progress subscriber
            results_subscriber (MessageSubscriberIF[EvaluationResultBatch]): The results subscriber
            global_rank (int): The global rank of the current process
            local_rank (int): The local rank of the current process on the current node

        Returns:
            tuple[MessagePublisher[EvaluationResultBatch], MessagePublisher[ProgressUpdate]]: The evaluation
                result publisher and the progress publisher
        """
        message_broker = MessageBroker()
        progress_publisher = MessagePublisher[ProgressUpdate](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        evaluation_result_publisher = MessagePublisher[EvaluationResultBatch](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )

        message_broker.add_subscriber(subscription=MessageTypes.EVALUATION_RESULT, subscriber=results_subscriber)
        message_broker.add_subscriber(
            subscription=MessageTypes.BATCH_PROGRESS_UPDATE,
            subscriber=progress_subscriber,
        )

        return evaluation_result_publisher, progress_publisher


if __name__ == "__main__":
    main()
