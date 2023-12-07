import logging
from pathlib import Path
from typing import Dict

import click
import click_pathlib
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import SaveMostRecentEpochOnlyCheckpointingStrategy
from llm_gym.config.config import AppConfig, DataLoaderConfig
from llm_gym.dataloader.create_index import create_memmap_index
from llm_gym.dataloader.create_packed_data import create_packed_data
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.fsdp.fsdp_runner import Runner
from llm_gym.gym import Gym
from llm_gym.logging_broker.message_broker import MessageBroker
from llm_gym.logging_broker.messages import BatchProgressUpdate, MessageTypes
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.logging_broker.subscriber_impl.batch_progress_subscriber import (
    DummyProgressSubscriber,
    RichProgressSubscriber,
)
from llm_gym.logging_broker.subscriber_impl.results_subscriber import RichResultSubscriber
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
@click.argument("model_path", type=str)
@click.argument("config_path", type=str)
@click.option(
    "--tokenizer_file",
    type=str,
    show_default=True,
    default="./data/tokenizer/tokenizer.json",
    help="path to tokenizer json",
)
@click.option("--max_new_tokens", type=int, show_default=True, default=200, help="maximum amount of tokens to generate")
@click.option("--chat", is_flag=True, show_default=True, default=False, help="activate 'chat' mode")
def entry_point_generate_text(model_path, config_path, tokenizer_file, max_new_tokens, chat):
    generate_text_main(model_path, config_path, tokenizer_file, max_new_tokens, chat)


@main.command(name="create_memmap_index")
@click.argument("src_path", type=str)
@click.option(
    "--index_path",
    type=str,
    default=None,
    help="output path for index. will use parent directory of src_path if none.",
)
def entry_point_create_memmap_index(src_path, index_path):
    create_memmap_index(src_path, index_path)


@main.command(name="create_packed_data")
@click.argument("src_path", type=str)
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
    "--jq_pattern",
    type=str,
    show_default=True,
    default=".text",
    help="jq pattern to extract the data from the json line.",
)
def entry_point_create_packed_data(src_path, dst_path, index_path, jq_pattern):
    create_packed_data(src_path, dst_path, index_path=index_path, jq_pattern=jq_pattern)


def load_app_config_dict(config_file_path: Path) -> Dict:
    cfg = OmegaConf.load(config_file_path)
    logging.info(f"Config\n {OmegaConf.to_yaml(cfg, resolve=True)}")
    return OmegaConf.to_container(cfg, resolve=True)


class Main:
    def __init__(self, config: AppConfig) -> None:
        # Checks whether this process was launched with ``torch.distributed.elastic``
        dist_launched = dist.is_torchelastic_launched()
        self.config = config

        self.experiment_id = get_date_of_run()

        self.resolvers = ResolverRegister(config=config)

        self.model: torch.nn.Module = self.resolvers.build_component_by_config(config=config.model)

        runner: Runner = self.resolvers.build_component_by_config(config=config.runner)

        self.wrapped_model = runner.wrap(model=self.model, local_rank=config.training.local_rank)

        self.optimizer: torch.optim.Optimizer = self.resolvers.build_component_by_config(
            config=config.optimizer, extra_kwargs=dict(params=self.wrapped_model.parameters())
        )

        self.scheduler = self.resolvers.build_component_by_config(
            config=config.scheduler, extra_kwargs=dict(optimizer=self.optimizer)
        )

        self.loss_fun: Loss = self.resolvers.build_component_by_config(config=config.loss)

        self.train_dataloader = self._create_dataloader(config=config.training.train_dataloader)
        validation_dataloader_lookup = {
            dataloader_tag: self._create_dataloader(config=config)
            for dataloader_tag, config in config.training.evaluation_dataloaders.items()
        }
        self.val_dataloader = validation_dataloader_lookup["val"]
        self.test_dataloader = validation_dataloader_lookup["test"]

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

        eval_split_lengths = {
            self.val_dataloader.dataloader_tag: len(self.val_dataloader) * config.training.world_size,
            self.test_dataloader.dataloader_tag: len(self.test_dataloader) * config.training.world_size,
        }
        train_split_lengths = {self.train_dataloader.dataloader_tag: len(self.train_dataloader)}

        # TODO: make this instantiation of subscribers configurable via config.yml and use "build_component_by_config"
        if not dist_launched or (dist_launched and dist.get_rank() == 0):
            progress_subscriber = RichProgressSubscriber(
                num_ranks=config.training.world_size,
                train_split_lengths=train_split_lengths,
                eval_split_lengths=eval_split_lengths,
            )
            evaluation_result_subscriber = RichResultSubscriber(num_ranks=config.training.world_size)
            message_broker.add_subscriber(
                subscription=MessageTypes.EVALUATION_RESULT, subscriber=evaluation_result_subscriber
            )

        else:
            progress_subscriber = DummyProgressSubscriber()
        message_broker.add_subscriber(
            subscription=MessageTypes.BATCH_PROGRESS_UPDATE,
            subscriber=progress_subscriber,
        )

        # Checkpointing
        config.checkpoint.dir_path.mkdir(parents=True, exist_ok=True)
        checkpointing_strategy = SaveMostRecentEpochOnlyCheckpointingStrategy()
        checkpointing_execution = FSDPToDiscCheckpointing(
            checkpoint_path=config.checkpoint.dir_path,
            experiment_id=self.experiment_id,
            global_rank=config.training.global_rank,
            checkpointing_rank=config.checkpoint.checkpointing_rank,
        )
        checkpointing = Checkpointing(
            checkpointing_execution=checkpointing_execution,
            checkpointing_strategy=checkpointing_strategy,
            num_ranks=config.training.world_size,
        )

        # Trainer
        self.trainer = Trainer(
            local_rank=config.training.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Evaluator
        self.eval_data_loaders = [self.val_dataloader, self.test_dataloader]

        self.evaluator = Evaluator(
            local_rank=config.training.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Gym
        self.gym = Gym(
            checkpointing=checkpointing,
            trainer=self.trainer,
            evaluator=self.evaluator,
            model=self.wrapped_model,
            optimizer=self.optimizer,
            loss_fun=self.loss_fun,
        )

    def run(self):
        self.gym.run(
            num_batches_per_rank=self.config.training.num_batches_per_rank,
            eval_interval_in_batches=self.config.training.eval_interval_in_batches_per_rank,
            train_data_loader=self.train_dataloader,
            evaluation_data_loaders=self.eval_data_loaders,
        )

    def _create_dataloader(self, config: DataLoaderConfig) -> LLMDataLoader:
        dataset = self.resolvers.build_component_by_config(config=config.config.dataset)
        collator = self.resolvers.build_component_by_config(config=config.config.collate_fn)
        sampler = self.resolvers.build_component_by_config(
            config=config.config.sampler, extra_kwargs=dict(dataset=dataset)
        )
        created_dataloader = self.resolvers.build_component_by_config(
            config=config,
            extra_kwargs=dict(
                dataset=dataset,
                sampler=sampler,
                collate_fn=collator,
            ),
        )
        assert isinstance(
            created_dataloader, LLMDataLoader
        ), f"Dataloader Class must use the {LLMDataLoader.__name__}-Interface"
        return created_dataloader


if __name__ == "__main__":
    main()
