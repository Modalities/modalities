import logging
from pathlib import Path
from typing import Dict, List, Tuple

import click
import click_pathlib
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Optimizer
from transformers import GPT2TokenizerFast

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import SaveKMostRecentCheckpointsStrategy
from llm_gym.config.config import AppConfig
from llm_gym.config.lookup_types import TokenizerTypes
from llm_gym.dataloader.create_index import IndexGenerator
from llm_gym.dataloader.create_packed_data import PackedDataGenerator
from llm_gym.dataloader.dataloader_factory import DataloaderFactory
from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.fsdp.fsdp_running_env import RunningEnv
from llm_gym.gym import Gym
from llm_gym.logging_broker.message_broker import MessageBroker
from llm_gym.logging_broker.messages import BatchProgressUpdate, MessageTypes
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.logging_broker.subscriber_impl.batch_progress_subscriber import (
    DummyProgressSubscriber,
    RichProgressSubscriber,
)
from llm_gym.logging_broker.subscriber_impl.results_subscriber import WandBEvaluationResultSubscriber
from llm_gym.loss_functions import Loss
from llm_gym.resolver_register import ResolverRegister
from llm_gym.trainer import Trainer
from llm_gym.util import get_date_of_run
from llm_gym.utils.generate_text import main as generate_text_main


@click.group()
def main() -> None:
    pass


config_option = click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)


@main.command(name="run")
@config_option
def entry_point_run_llmgym(config_file_path: Path):
    config_dict = load_app_config_dict(config_file_path)
    config = AppConfig.model_validate(config_dict)
    main = Main(config)
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
    "--tokenizer_type",
    type=TokenizerTypes,
    show_default=True,
    default=GPT2TokenizerFast,
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
    type=str,
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
    cfg = OmegaConf.load(config_file_path)
    logging.info(f"Config\n {OmegaConf.to_yaml(cfg, resolve=True)}")
    return OmegaConf.to_container(cfg, resolve=True)


class Main:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # warmstart
        self.global_train_batch_id = 0
        self.warmstart_experiment_id = "2023-11-15-11:53:54_PM"

        self.experiment_id = get_date_of_run()
        self.resolvers = ResolverRegister(config=config)
        self.running_env: RunningEnv = self.resolvers.build_component_by_config(config=self.config.running_env)

    def run(self):
        with self.running_env as running_env:
            (
                gym,
                train_dataloader,
                eval_data_loaders,
                checkpointing,
                wrapped_model,
                optimizer,
            ) = self.construct_components(self.resolvers, self.config, running_env=running_env)

            gym.run(
                num_batches_per_rank=self.config.training.num_batches_per_rank,
                callback_interval_in_batches=self.config.training.callback_interval_in_batches_per_rank,
                train_data_loader=train_dataloader,
                evaluation_data_loaders=eval_data_loaders,
                checkpointing=checkpointing,
                model=wrapped_model,
                optimizer=optimizer,
            )

    def construct_components(
        self, resolvers: ResolverRegister, config: AppConfig, running_env: RunningEnv
    ) -> Tuple[Gym, LLMDataLoader, List[LLMDataLoader], Checkpointing, nn.Module, Optimizer]:
        train_dataloader = DataloaderFactory.get_dataloader(resolvers, config.training.train_dataloader)
        validation_dataloader_lookup = {
            dataloader_tag: DataloaderFactory.get_dataloader(resolvers, config=config)
            for dataloader_tag, config in config.training.evaluation_dataloaders.items()
        }
        # TODO: should get replaced with dynamic handling of multiple validation dataloaders
        val_dataloader = validation_dataloader_lookup["val"]
        test_dataloader = validation_dataloader_lookup["test"]

        # TODO: check why not *config.training.world_size
        #  and consider just using config.training.num_training_samples for progress Subscriber
        train_split_lengths = {train_dataloader.dataloader_tag: len(train_dataloader)}

        # Checkpointing
        checkpointing = self.get_checkpointing(config=config, running_env=running_env)

        wrapped_model, optimizer = self.get_model_and_optimizer(
            config=config, running_env=running_env, checkpointing=checkpointing
        )

        # Loss function
        loss_fun: Loss = self.resolvers.build_component_by_config(config=config.loss)

        # Logging

        eval_split_lengths = {
            val_dataloader.dataloader_tag: len(val_dataloader) * config.training.world_size,
            test_dataloader.dataloader_tag: len(test_dataloader) * config.training.world_size,
        }
        train_split_lengths = {train_dataloader.dataloader_tag: len(train_dataloader)}

        evaluation_result_publisher, batch_processed_publisher = self.get_logging_publishers(
            config=config, train_split_lengths=train_split_lengths, eval_split_lengths=eval_split_lengths
        )

        # Trainer
        trainer = Trainer(
            local_rank=config.training.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Evaluator
        evaluator = Evaluator(
            local_rank=config.training.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Gym
        gym = Gym(
            trainer=trainer,
            evaluator=evaluator,
            loss_fun=loss_fun,
        )
        return gym, train_dataloader, [val_dataloader, test_dataloader], checkpointing, wrapped_model, optimizer

    def get_model_and_optimizer(
        self, config: AppConfig, running_env: RunningEnv, checkpointing: Checkpointing
    ) -> Tuple[nn.Module, Optimizer]:
        model: torch.nn.Module = self.resolvers.build_component_by_config(config=config.model)

        if self.global_train_batch_id > 0:  # warm start
            wrapped_model = checkpointing.load_model_checkpoint(
                experiment_id=self.warmstart_experiment_id,
                global_train_batch_id=self.global_train_batch_id,
                model=model,
            )

            optimizer: torch.optim.Optimizer = self.resolvers.build_component_by_config(
                config=config.optimizer, extra_kwargs=dict(params=wrapped_model.parameters())
            )

            optimizer = checkpointing.load_optimizer_checkpoint(
                optimizer=optimizer,
                model=wrapped_model,
                experiment_id=self.warmstart_experiment_id,
                global_train_batch_id=self.global_train_batch_id,
            )

        else:
            wrapped_model = running_env.wrap_model(model=model, sync_module_states=False)
            optimizer: torch.optim.Optimizer = self.resolvers.build_component_by_config(
                config=config.optimizer, extra_kwargs=dict(params=wrapped_model.parameters())
            )
        # TODO implement scheduler
        #  scheduler = self.resolvers.build_component_by_config(
        #     config=config.scheduler, extra_kwargs=dict(optimizer=self.optimizer)
        #  )

        return wrapped_model, optimizer

    def get_checkpointing(self, config: AppConfig, running_env: RunningEnv) -> Checkpointing:
        checkpointing_strategy = SaveKMostRecentCheckpointsStrategy(k=-1)
        checkpointing_execution = FSDPToDiscCheckpointing(
            checkpoint_path=Path("/raid/s3/opengptx/max_lue/LLMgym/checkpoints"),
            experiment_id=self.experiment_id,
            global_rank=config.training.global_rank,
            checkpointing_rank=0,
            model_wrapping_fn=running_env.wrap_model,
        )
        checkpointing = Checkpointing(
            checkpointing_execution=checkpointing_execution,
            checkpointing_strategy=checkpointing_strategy,
            num_ranks=config.training.world_size,
        )
        return checkpointing

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

        if config.training.global_rank == 0:
            progress_subscriber = RichProgressSubscriber(
                num_ranks=config.training.world_size,
                train_split_lengths=train_split_lengths,
                eval_split_lengths=eval_split_lengths,
            )
            evaluation_result_subscriber = WandBEvaluationResultSubscriber(
                num_ranks=config.training.world_size,
                project=config.wandb.project_name,
                experiment_id=self.experiment_id,
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
