#!/usr/bin/env python

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Type

import click
import click_pathlib
from pydantic import BaseModel, FilePath

from modalities.activation_checkpointing import apply_activation_checkpointing_inplace
from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.config.instantiation_models import (
    PackedDatasetComponentsInstantiationModel,
    TrainingComponentsInstantiationModel,
)
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.inference.inference import generate_text
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import BatchProgressUpdate, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.trainer import Trainer
from modalities.util import get_total_number_of_trainable_parameters, print_rank_0


@click.group()
def main() -> None:
    pass


@main.command(name="run")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)
def entry_point_run_modalities(config_file_path: Path):
    main_obj = Main(config_file_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main_obj.run(components)


@main.command(name="generate_text")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)
def entry_point_generate_text(config_file_path: FilePath):
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
def entry_point_convert_pytorch_to_hf_checkpoint(
    config_file_path: Path, output_hf_checkpoint_dir: Path, prediction_key: str
) -> HFModelAdapter:
    cp = CheckpointConversion(config_file_path, output_hf_checkpoint_dir)
    hf_model = cp.convert_pytorch_to_hf_checkpoint(prediction_key=prediction_key)
    print(f"Model was successfully converted and saved to {output_hf_checkpoint_dir}")
    return hf_model


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
def entry_point_data_create_raw_index(src_path, index_path):
    """
    Utility for indexing a large jsonl-file's content.
    Background is the ability to further process the respective file without loading it,
    while splitting its content line-based. This step is necessary in advance of further processing like tokenization.
    It is only necessary once for a jsonl-file and allows therefore different tokenizations without re-indexing.
    """
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        raise ValueError("index already exists. delete it or specify different output folder.")

    print(f"reading raw data from {src_path}")
    print(f"writing index to {index_path}")
    generator = IndexGenerator(src_path)
    generator.create_index(index_path)


@data.command(name="pack_encoded_data")
@click.argument("config_path", type=FilePath)
def entry_point_pack_encoded_data(config_path: FilePath):
    """
    Utility to encode an indexed, large jsonl-file.

    (see also `create_index` for more information)
    Returns .pbin-file, which can be inserted into a training process directly
    and does not require its original jsonl-file or the respective index file anymore.
    """
    # TODO: if we want to use alternative entrypoints together with the ResolverRegistry,
    #  we can currently not rely on the existing class resolver.
    #  This is based on its connection to the overall `AppConfig`.
    #  One would requires an object of it to instantiate the ResolverRegistry.
    #  This could get resolved by implementing on own ResolverRegistry for each entrypoint or adapting the existing
    #  ResolverRegistry to work dynamically with any type-hinted config object from config.py.
    config = load_app_config_dict(config_path)
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components: PackedDatasetComponentsInstantiationModel = component_factory.build_components(
        config_dict=config, components_model_type=PackedDatasetComponentsInstantiationModel
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


@data.command(name="merge_packed_data")
@click.argument("src_paths", type=click.types.Path(exists=True, path_type=Path), nargs=-1, required=True)
@click.argument("target_path", type=click.types.Path(file_okay=False, dir_okay=False, path_type=Path))
def entry_point_merge_packed_data(src_paths, target_path):
    """
    Utility for merging different pbin-files into one.
    This is especially useful, if different datasets were at different points in time or if one encoding takes so long,
    that the overall process was done in chunks.
    It is important that the same tokenizer got used for all chunks.

    Specify an arbitrary amount of pbin-files and/or directory containing such as input.
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


class Main:
    def __init__(self, config_path: Path) -> None:
        self.config_dict = load_app_config_dict(config_path)
        self.config_path = config_path

        self.registry = Registry(COMPONENTS)
        self.component_factory = ComponentFactory(registry=self.registry)

    def add_custom_component(self, component_key: str, variant_key: str, custom_component, custom_config) -> None:
        self.registry.add_entity(
            component_key=component_key,
            variant_key=variant_key,
            component_type=custom_component,
            component_config_type=custom_config,
        )

    def build_components(self, components_model_type: Type[BaseModel]) -> BaseModel:
        components = self.component_factory.build_components(
            config_dict=self.config_dict, components_model_type=components_model_type
        )
        return components

    def run(self, components: TrainingComponentsInstantiationModel):
        print_rank_0(f"Initialize Model at {datetime.now()}.")
        # save the config file to the checkpointing path
        if components.settings.cuda_env.global_rank == 0:
            experiment_path = components.settings.paths.checkpointing_path / components.settings.experiment_id
            os.makedirs(experiment_path, exist_ok=True)
            shutil.copy(self.config_path, experiment_path / self.config_path.name)

        evaluation_result_publisher, batch_processed_publisher = self.get_logging_publishers(
            progress_subscriber=components.batch_progress_subscriber,
            results_subscriber=components.evaluation_subscriber,
            global_rank=components.settings.cuda_env.global_rank,
            local_rank=components.settings.cuda_env.local_rank,
        )

        # Trainer
        global_num_tokens_per_train_step = (
            components.settings.training.local_train_micro_batch_size
            * components.settings.training.sequence_length
            * components.settings.training.gradient_acc_steps
            * components.settings.cuda_env.world_size
        )
        trainer = Trainer(
            global_rank=components.settings.cuda_env.global_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
            gradient_acc_steps=components.settings.training.gradient_acc_steps,
            gradient_clipper=components.gradient_clipper,
            global_num_tokens_per_train_step=global_num_tokens_per_train_step,
        )

        # Evaluator
        evaluator = Evaluator(
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Gym
        gym = Gym(
            trainer=trainer,
            evaluator=evaluator,
            loss_fun=components.loss_fn,
            num_ranks=components.settings.cuda_env.world_size,
        )
        wrapped_model = components.wrapped_model
        num_params = get_total_number_of_trainable_parameters(wrapped_model)
        components.evaluation_subscriber.consume_dict({"No. Parameters": num_params})
        logging.info(f"Training model with {num_params} parameters.")

        if len(components.settings.training.activation_checkpointing_modules) > 0:
            apply_activation_checkpointing_inplace(
                model=wrapped_model,
                activation_checkpointing_modules=components.settings.training.activation_checkpointing_modules,
            )
        print_rank_0(f"Model initialized at {datetime.now()}.")

        gym.run(
            train_data_loader=components.train_dataloader,
            evaluation_data_loaders=components.eval_dataloaders,
            checkpoint_saving=components.checkpoint_saving,
            model=wrapped_model,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            checkpointing_interval_in_steps=components.settings.training.checkpointing_interval_in_steps,
            evaluation_interval_in_steps=components.settings.training.evaluation_interval_in_steps,
            training_log_interval_in_steps=components.settings.training.training_log_interval_in_steps,
        )

    def get_logging_publishers(
        self,
        progress_subscriber: MessageSubscriberIF[BatchProgressUpdate],
        results_subscriber: MessageSubscriberIF[EvaluationResultBatch],
        global_rank: int,
        local_rank: int,
    ) -> Tuple[MessagePublisher[EvaluationResultBatch], MessagePublisher[BatchProgressUpdate],]:
        message_broker = MessageBroker()
        batch_processed_publisher = MessagePublisher[BatchProgressUpdate](
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

        return evaluation_result_publisher, batch_processed_publisher


if __name__ == "__main__":
    main()
