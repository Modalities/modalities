import logging
from pathlib import Path
from typing import Callable, Dict

import click
import click_pathlib
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import SaveMostRecentEpochOnlyCheckpointingStrategy
from llm_gym.config.config import AppConfig
from llm_gym.dataloader.create_index import create_memmap_index
from llm_gym.dataloader.create_packed_data import create_packed_data
from llm_gym.dataloader.dataset import Dataset, DatasetSplit, MemMapDataset, PackedDataset  # noqa: F401
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
from llm_gym.models.gpt2.collator import GPT2LLMCollator
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

        self.dataset_path = config.data.dataset_dir_path

        resolvers = ResolverRegister(config=config)

        self.model: torch.nn.Module = resolvers.build_component_by_config(config=config.model)

        runner: Runner = resolvers.build_component_by_config(config=config.runner)

        self.wrapped_model = runner.wrap(model=self.model, local_rank=config.training.local_rank)

        self.optimizer: torch.optim.Optimizer = resolvers.build_component_by_config(
            config=config.optimizer, extra_kwargs=dict(params=self.wrapped_model.parameters())
        )

        self.scheduler = resolvers.build_component_by_config(
            config=config.scheduler, extra_kwargs=dict(optimizer=self.optimizer)
        )

        self.loss_fun: Loss = resolvers.build_component_by_config(config=config.loss)

        # Create instances
        dataset_split = self.create_datasplit(config=config)

        # Create samplers
        sampler_splits = self.create_samplers(dataset_split)

        collator = GPT2LLMCollator(
            sample_key=config.data.sample_key,
            target_key=config.data.target_key,
        )

        dataloader_splits = self.create_dataloaders(
            dataset_split,
            train_sampler=sampler_splits["train"],
            val_sampler=sampler_splits["val"],
            test_sampler=sampler_splits["test"],
            collate_fn=collator,
        )

        self.train_dataloader = dataloader_splits["train"]
        self.val_dataloader = dataloader_splits["val"]
        self.test_dataloader = dataloader_splits["test"]

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
            self.val_dataloader.dataset_tag: len(self.val_dataloader) * config.training.world_size,
            self.test_dataloader.dataset_tag: len(self.test_dataloader) * config.training.world_size,
        }
        train_split_lengths = {self.train_dataloader.dataset_tag: len(self.train_dataloader)}

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
            eval_interval_in_batches=self.config.training.eval_interval_per_rank,
            train_data_loader=self.train_dataloader,
            evaluation_data_loaders=self.eval_data_loaders,
        )

    def create_datasplit(self, config: AppConfig) -> DatasetSplit:
        # on the fly tokenization
        # TODO: centralize this used Tokenizer
        # TODO: consider using instantiation from config and unify this with the parallel llm_gym.data implementation
        return Dataset.from_path(
            config.data.dataset_dir_path,
            target_dataset_cls=MemMapDataset,
            split_size=(0.999, 0.0005, 0.0005),  # PackedDataset, MemMapDataset
        )

    def create_samplers(self, dataset_split: DatasetSplit) -> Dict[str, DistributedSampler]:
        sampler_splits = dict()

        sampler_splits["train"] = DistributedSampler(
            dataset=dataset_split.train,
            rank=self.config.training.global_rank,
            num_replicas=self.config.training.world_size,
            shuffle=True,
        )

        sampler_splits["val"] = DistributedSampler(
            dataset=dataset_split.validation,
            rank=self.config.training.global_rank,
            num_replicas=self.config.training.world_size,
        )

        sampler_splits["test"] = DistributedSampler(
            dataset=dataset_split.test,
            rank=self.config.training.global_rank,
            num_replicas=self.config.training.world_size,
        )

        return sampler_splits

    def create_dataloaders(
        self,
        dataset_split: DatasetSplit,
        train_sampler: DistributedSampler,
        val_sampler: DistributedSampler,
        test_sampler: DistributedSampler,
        collate_fn: Callable,
    ) -> Dict[str, LLMDataLoader]:
        """Create dataset splits."""

        data_loader_splits = {}

        # create dataloaders
        collate_fn = GPT2LLMCollator(
            sample_key=self.config.data.sample_key,
            target_key=self.config.data.target_key,
        )
        data_loader_splits["train"] = LLMDataLoader(
            dataset=dataset_split.train,
            dataset_tag=self.config.data.dataloader.train_dataset_tag,
            batch_size=self.config.training.training_batch_size,
            sampler=train_sampler,
            **self.config.data.dataloader.cuda_kwargs.model_dump(),
            collate_fn=collate_fn,
        )
        data_loader_splits["val"] = LLMDataLoader(
            dataset=dataset_split.validation,
            dataset_tag=self.config.data.dataloader.val_dataset_tag,
            batch_size=self.config.training.evaluation_batch_size,
            sampler=val_sampler,
            **self.config.data.dataloader.cuda_kwargs.model_dump(),
            collate_fn=collate_fn,
        )
        data_loader_splits["test"] = LLMDataLoader(
            dataset=dataset_split.test,
            dataset_tag=self.config.data.dataloader.test_dataset_tag,
            batch_size=self.config.training.test_batch_size,
            sampler=test_sampler,
            **self.config.data.dataloader.cuda_kwargs.model_dump(),
            collate_fn=collate_fn,
        )

        return data_loader_splits


if __name__ == "__main__":
    main()
