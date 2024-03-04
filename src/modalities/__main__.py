#!/usr/bin/env python

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import click
import click_pathlib

from modalities.activation_checkpointing import apply_activation_checkpointing_inplace
from modalities.batch import EvaluationResultBatch
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ComponentsModel, ProcessGroupBackendType, TokenizerTypes, load_app_config_dict
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import PackedDataGenerator
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import BatchProgressUpdate, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.trainer import Trainer
from modalities.util import compute_number_of_trainable_parameters, get_callback_interval_in_batches_per_rank
from modalities.utils.generate_text import main as generate_text_main


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
    config_dict = load_app_config_dict(config_file_path)
    main = Main(config_dict, config_file_path)
    main.run()


@main.command(name="generate_text")
@click.argument("model_path", type=Path)
@click.argument("config_path", type=Path)
@click.option(
    "--tokenizer_type",
    type=TokenizerTypes,
    show_default=True,
    default=TokenizerTypes.GPT2TokenizerFast,
    help="Specify which Tokenizer (inheriting from transformers.PretrainedTokenizers) should get used.",
)
@click.option(
    "--tokenizer_file",
    type=Path,
    show_default=True,
    default=Path(__file__).parents[2] / Path("data/tokenizer/tokenizer.json"),
    help="path to tokenizer json",
)
@click.option("--max_new_tokens", type=int, show_default=True, default=200, help="maximum amount of tokens to generate")
@click.option("--chat", is_flag=True, show_default=True, default=False, help="activate 'chat' mode")
def entry_point_generate_text(model_path, config_path, tokenizer_type, tokenizer_file, max_new_tokens, chat):
    tokenizer = tokenizer_type.value(tokenizer_file=str(tokenizer_file))
    generate_text_main(model_path, config_path, tokenizer, max_new_tokens, chat)


@main.command(name="create_memmap_index")
@click.argument("src_path", type=Path)
@click.option(
    "--index_path",
    type=Path,
    default=None,
    help="output path for index. will use parent directory of src_path if none.",
)
def entry_point_create_memmap_index(src_path, index_path):
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        raise ValueError("index already exists. delete it or specify different output folder.")

    print(f"reading raw data from {src_path}")
    print(f"writing index to {index_path}")
    generator = IndexGenerator(src_path)
    generator.create_index(index_path)


@main.command(name="create_packed_data")
@click.argument("src_path", type=Path)
@click.option(
    "--dst_path",
    type=str,
    default=None,
    help="output path for packed data file. will use parent directory of src_path if none.",
)
@click.option(
    "--index_path",
    type=Path,
    default=None,
    help="input path for index. will search in parent directory of src_path if none.",
)
@click.option(
    "--tokenizer_type",
    type=TokenizerTypes,
    show_default=True,
    default=TokenizerTypes.GPT2TokenizerFast,
    help="Specify which Tokenizer (inheriting from transformers.PretrainedTokenizers) should get used.",
)
@click.option(
    "--tokenizer_file",
    type=Path,
    show_default=True,
    default=Path(__file__).parents[2] / Path("data/tokenizer/tokenizer.json"),
    help="path to tokenizer json",
)
@click.option(
    "--jq_pattern",
    type=str,
    show_default=True,
    default=".text",
    help="jq pattern to extract the data from the json line.",
)
def entry_point_create_packed_data(src_path, dst_path, index_path, tokenizer_type, tokenizer_file, jq_pattern):
    # TODO: if we want to use alternative entrypoints together with the ResolverRegistry,
    #  we can currently not rely on the existing class resolver.
    #  This is based on its connection to the overall `AppConfig`.
    #  One would requires an object of it to instantiate the ResolverRegistry.
    #  This could get resolved by implementing on own ResolverRegistry for each entrypoint or adapting the existing
    #  ResolverRegistry to work dynamically with any type-hinted config object from config.py.
    tokenizer = tokenizer_type.value(tokenizer_file=str(tokenizer_file))
    generator = PackedDataGenerator(src_path, index_path=index_path, tokenizer=tokenizer, jq_pattern=jq_pattern)
    generator.run(dst_path)


class Main:
    def __init__(self, config_dict: Dict, config_path: Path) -> None:
        self.config_dict = config_dict
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

    def run(self):
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            components: ComponentsModel = self.component_factory.build_components(
                config_dict=self.config_dict, components_model_type=ComponentsModel
            )

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
            trainer = Trainer(
                local_rank=components.settings.cuda_env.local_rank,
                batch_progress_publisher=batch_processed_publisher,
                evaluation_result_publisher=evaluation_result_publisher,
                gradient_acc_steps=components.settings.training.gradient_acc_steps,
            )

            # Evaluator
            evaluator = Evaluator(
                local_rank=components.settings.cuda_env.local_rank,
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
            logging.info(f"Training model with {compute_number_of_trainable_parameters(wrapped_model)} parameters.")

            if components.settings.training.do_apply_activation_checkpointing:
                apply_activation_checkpointing_inplace(wrapped_model)

            callback_interval_in_batches_per_rank = get_callback_interval_in_batches_per_rank(
                callback_interval_in_samples=components.settings.training.callback_interval_in_samples,
                local_train_micro_batch_size=components.settings.training.local_train_micro_batch_size,
                gradient_acc_steps=components.settings.training.gradient_acc_steps,
                world_size=components.settings.cuda_env.world_size,
            )

            gym.run(
                callback_interval_in_batches=callback_interval_in_batches_per_rank,
                train_data_loader=components.train_dataloader,
                evaluation_data_loaders=components.eval_dataloaders,
                checkpointing=components.checkpointing,
                model=wrapped_model,
                optimizer=components.optimizer,
            )
            print("done")

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
