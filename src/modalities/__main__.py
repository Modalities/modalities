#!/usr/bin/env python

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Type

import click
import click_pathlib
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Optimizer

from modalities.activation_checkpointing import apply_activation_checkpointing_inplace
from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpointing import Checkpointing, CheckpointingIF
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import AppConfig, ModalitiesSetupConfig, RunMode
from modalities.config.lookup_types import TokenizerTypes
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import PackedDataGenerator
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import BatchProgressUpdate, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber_impl.batch_progress_subscriber import (
    DummyProgressSubscriber,
    RichProgressSubscriber,
)
from modalities.logging_broker.subscriber_impl.results_subscriber import WandBEvaluationResultSubscriber
from modalities.loss_functions import Loss
from modalities.registry.registry_factory import RegistryFactory
from modalities.running_env.fsdp.fsdp_running_env import RunningEnv
from modalities.trainer import Trainer
from modalities.util import compute_number_of_trainable_parameters, get_date_of_run
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
    main = Main(config_dict)
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


def load_app_config_dict(config_file_path: Path) -> Dict:
    int_env_variable_names = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]

    def resolver_fun(var_name: str) -> int:
        return int(os.getenv(var_name)) if var_name in int_env_variable_names else os.getenv(var_name)

    OmegaConf.register_new_resolver("modalities_env", resolver_fun)

    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_dict


class Main:
    def __init__(
        self, config_dict: Dict, component_names: List[str] = None, custom_config_types: List[Type] = None
    ) -> None:
        self.config_dict = config_dict
        self.experiment_id = get_date_of_run()
        self.custom_config_types = custom_config_types if custom_config_types is not None else []
        self.component_names = (
            component_names
            if component_names is not None
            else ["running_env", "loss", "checkpointing", "model", "optimizer", "tokenizer", "dataset"]
        )

        component_registry = RegistryFactory.get_component_registry()
        component_config_registry = RegistryFactory.get_config_registry()
        self.component_factory = ComponentFactory(
            config_registry=component_config_registry, component_registry=component_registry
        )

    def build_component_dict(self) -> Dict:
        component_dict = self.component_factory.build_config(
            config_dict=self.config_dict, component_names=self.component_names
        )
        return component_dict

    def run(self):
        self.build_component_dict()

        self.running_env: RunningEnv = self.resolvers.build_component_by_config(config=self.config.running_env)
        with self.running_env as running_env:
            (
                gym,
                train_dataloader,
                eval_data_loaders,
                checkpointing,
                wrapped_model,
                optimizer,
            ) = self.construct_components(resolvers=self.resolvers, config=self.config, running_env=running_env)

            logging.info(f"Training model with {compute_number_of_trainable_parameters(wrapped_model)} parameters.")

            gym.run(
                callback_interval_in_batches=self.config.training.callback_interval_in_batches_per_rank,
                train_data_loader=train_dataloader,
                evaluation_data_loaders=eval_data_loaders,
                checkpointing=checkpointing,
                model=wrapped_model,
                optimizer=optimizer,
            )

    def construct_components(
        self, resolvers, config: AppConfig, running_env: RunningEnv
    ) -> Tuple[Gym, LLMDataLoader, List[LLMDataLoader], CheckpointingIF, nn.Module, Optimizer]:
        # Checkpointing

        checkpointing = None  # TODO pass in the component

        # Model and optimizer
        wrapped_model, optimizer = self.get_model_and_optimizer(
            config=config, running_env=running_env, checkpointing=checkpointing
        )
        if config.training.do_apply_activation_checkpointing:
            apply_activation_checkpointing_inplace(wrapped_model)
            logging.info("Applied activation checkpointing!")

        # Loss function
        loss_fun: Loss = resolvers.build_component_by_config(config=config.loss)

        # Dataloaders
        # skip_num_samples = 0
        # if run_mode == RunMode.WARM_START:
        #     skip_num_samples = config.modalities_setup.settings.checkpoint_num_seen_samples

        skip_num_local_train_batches = config.training.skip_num_local_train_batches
        train_dataloader = DataloaderFactory.get_dataloader(
            resolvers=resolvers, config=config.data.train_dataloader, skip_num_batches=skip_num_local_train_batches
        )
        eval_dataloaders = [
            DataloaderFactory.get_dataloader(resolvers=resolvers, config=dataloader_config)
            for dataloader_config in config.data.eval_dataloaders
        ]

        # Logging
        eval_split_lengths = {
            dataloader.dataloader_tag: len(dataloader) * config.training.world_size * dataloader.sampler_batch_size
            for dataloader in eval_dataloaders
        }

        # TODO: check why not *config.training.world_size
        #  and consider just using config.training.num_training_samples for progress Subscriber
        train_split_lengths = {
            train_dataloader.dataloader_tag: (len(train_dataloader) + skip_num_local_train_batches)
            * config.training.world_size
            * train_dataloader.sampler_batch_size
        }

        evaluation_result_publisher, batch_processed_publisher = self.get_logging_publishers(
            config=config, train_split_lengths=train_split_lengths, eval_split_lengths=eval_split_lengths
        )

        # Trainer
        trainer = Trainer(
            local_rank=config.training.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
            gradient_acc_step=config.training.gradient_acc_step,
        )

        # Evaluator
        evaluator = Evaluator(
            local_rank=config.training.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Gym
        gym = Gym(trainer=trainer, evaluator=evaluator, loss_fun=loss_fun, num_ranks=config.training.world_size)

        return gym, train_dataloader, eval_dataloaders, checkpointing, wrapped_model, optimizer

    def get_model_and_optimizer(
        self, config: AppConfig, running_env: RunningEnv, checkpointing: Checkpointing
    ) -> Tuple[nn.Module, Optimizer]:
        run_mode = config.modalities_setup.run_mode

        model: torch.nn.Module = self.resolvers.build_component_by_config(config=config.model)

        if run_mode == RunMode.WARM_START:
            warm_start_settings: ModalitiesSetupConfig.WarmStartSettings = config.modalities_setup.settings
            wrapped_model = checkpointing.load_model_checkpoint(
                file_path=warm_start_settings.checkpoint_model_path,
                model=model,
            )

            optimizer: torch.optim.Optimizer = self.resolvers.build_component_by_config(
                config=config.optimizer, extra_kwargs=dict(params=wrapped_model.parameters())
            )

            # TODO improve this
            if warm_start_settings.checkpoint_optimizer_path is None:
                raise (
                    NotImplementedError(
                        "So far we always have to provide an optimizer checkpoint. "
                        "For fine-tuning a pre-trained, we might not want to load "
                        "an optimizer checkpoint."
                    )
                )

            optimizer = checkpointing.load_optimizer_checkpoint(
                optimizer=optimizer, model=wrapped_model, file_path=warm_start_settings.checkpoint_optimizer_path
            )

        else:
            wrapped_model = running_env.wrap_model(model=model, sync_module_states=False)
            optimizer: torch.optim.Optimizer = self.resolvers.build_component_by_config(
                config=config.optimizer, extra_kwargs=dict(params=wrapped_model.parameters())
            )

        # TODO implement scheduler
        # scheduler = self.resolvers.build_component_by_config(
        #     config=config.scheduler, extra_kwargs=dict(optimizer=self.optimizer)
        # )

        return wrapped_model, optimizer

    def get_logging_publishers(
        self, config: AppConfig, train_split_lengths: Dict[str, int], eval_split_lengths: Dict[str, int]
    ) -> Tuple[MessagePublisher[EvaluationResultBatch], MessagePublisher[BatchProgressUpdate],]:
        # Message Broker
        message_broker = MessageBroker()
        batch_processed_publisher = MessagePublisher[BatchProgressUpdate](
            message_broker=message_broker,
            global_rank=config.training.global_rank,
            local_rank=config.training.local_rank,
        )
        evaluation_result_publisher = MessagePublisher[EvaluationResultBatch](
            message_broker=message_broker,
            global_rank=config.training.global_rank,
            local_rank=config.training.local_rank,
        )

        # TODO make logging rank configurable
        # TODO: make this instantiation of subscribers configurable via config.yml and use "build_component_by_config"
        if config.training.global_rank == 0:
            progress_subscriber = RichProgressSubscriber(
                num_ranks=config.training.world_size,
                train_split_num_samples=train_split_lengths,
                eval_splits_num_samples=eval_split_lengths,
            )
            evaluation_result_subscriber = WandBEvaluationResultSubscriber(
                num_ranks=config.training.world_size,
                project=config.wandb.project_name,
                experiment_id=self.experiment_id,
                mode=config.wandb.mode,
                dir=config.wandb.dir,
                experiment_config=config,
            )
            message_broker.add_subscriber(
                subscription=MessageTypes.EVALUATION_RESULT, subscriber=evaluation_result_subscriber
            )

        else:
            progress_subscriber = DummyProgressSubscriber()
        message_broker.add_subscriber(
            subscription=MessageTypes.BATCH_PROGRESS_UPDATE,
            subscriber=progress_subscriber,
        )

        return evaluation_result_publisher, batch_processed_publisher


if __name__ == "__main__":
    main()
