import logging
from pathlib import Path
from typing import Callable, Dict

import click
import click_pathlib
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import SaveMostRecentEpochOnlyCheckpointingStrategy
from llm_gym.config.config import AppConfig
from llm_gym.data.instances import TextInstances
from llm_gym.data.mmap_dataset import make_dataset
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
from llm_gym.logging_broker.subscriber_impl.results_subscriber import WandBEvaluationResultSubscriber
from llm_gym.loss_functions import Loss
from llm_gym.models.gpt2.collator import GPT2LLMCollator
from llm_gym.resolver_register import ResolverRegister
from llm_gym.trainer import Trainer
from llm_gym.util import get_date_of_run


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
        instance_splits = self.create_instances(config=config)

        # Create samplers
        sampler_splits = self.create_samplers(
            train_instances=instance_splits["train"],
            val_instances=instance_splits["val"],
            test_instances=instance_splits["test"],
        )

        collator = GPT2LLMCollator(
            sample_key=config.data.sample_key,
            target_key=config.data.target_key,
        )

        dataloader_splits = self.create_dataloaders(
            train_instances=instance_splits["train"],
            val_instances=instance_splits["val"],
            test_instances=instance_splits["test"],
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

        if not dist_launched or (dist_launched and dist.get_rank() == 0):
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

    def create_instances(self, config: AppConfig) -> Dict[str, TextInstances]:
        # on the fly tokenization
        # from llm_gym.dataloader.dataset import Dataset as Dataset_Wrapper
        # from llm_gym.dataloader.dataset import MemMapDataset
        # from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader
        # from transformers import GPT2TokenizerFast

        # reader = LargeFileLinesReader(config.data.dataset_dir_path, lazy_init=True)
        # tokenizer = GPT2TokenizerFast(tokenizer_file="./data/tokenizer/tokenizer.json")
        # dataset_dict = Dataset_Wrapper.from_reader(
        #     reader, target_dataset_cls=MemMapDataset, split_size=(0.999, 0.0005, 0.0005), tokenizer=tokenizer
        # )  # noqa: E501
        # instance_splits = dict()
        # instance_splits["train"] = dataset_dict.train
        # instance_splits["val"] = dataset_dict.validation
        # instance_splits["test"] = dataset_dict.test
        # return instance_splits

        dataset_path = config.data.dataset_dir_path
        sequence_len = config.data.sequence_len
        instance_splits = dict()

        for partition in ["train", "val", "test"]:
            dataset_filename_prefix = list(
                set([dataset_path.joinpath(filename.stem) for filename in dataset_path.glob(f"*{partition}*.bin")])
            )[0]
            text_dataset = make_dataset(path=dataset_filename_prefix)
            num_samples = config.training.num_training_batches * config.training.training_batch_size
            instances = TextInstances(
                sample_key=config.data.sample_key,
                text_dataset=text_dataset,
                doc_idx=np.arange(0, len(text_dataset)),
                dataset_dir=dataset_path,
                num_samples=num_samples,
                dataset_name=partition,
                sequence_len=sequence_len,
            )
            instance_splits[partition] = instances

        return instance_splits

    def create_samplers(
        self,
        train_instances: TextInstances,
        val_instances: TextInstances,
        test_instances: TextInstances,
    ) -> Dict[str, DistributedSampler]:
        sampler_splits = dict()

        sampler_splits["train"] = DistributedSampler(
            dataset=train_instances,
            rank=self.config.training.global_rank,
            num_replicas=self.config.training.world_size,
            shuffle=True,
        )

        sampler_splits["val"] = DistributedSampler(
            dataset=val_instances,
            rank=self.config.training.global_rank,
            num_replicas=self.config.training.world_size,
        )

        sampler_splits["test"] = DistributedSampler(
            dataset=test_instances,
            rank=self.config.training.global_rank,
            num_replicas=self.config.training.world_size,
        )

        return sampler_splits

    def create_dataloaders(
        self,
        train_instances: Dataset,
        val_instances: Dataset,
        test_instances: Dataset,
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
            dataset=train_instances,
            dataset_tag=self.config.data.dataloader.train_dataset_tag,
            batch_size=self.config.training.training_batch_size,
            sampler=train_sampler,
            **self.config.data.dataloader.cuda_kwargs.model_dump(),
            collate_fn=collate_fn,
        )
        data_loader_splits["val"] = LLMDataLoader(
            dataset=val_instances,
            dataset_tag=self.config.data.dataloader.val_dataset_tag,
            batch_size=self.config.training.evaluation_batch_size,
            sampler=val_sampler,
            **self.config.data.dataloader.cuda_kwargs.model_dump(),
            collate_fn=collate_fn,
        )
        data_loader_splits["test"] = LLMDataLoader(
            dataset=test_instances,
            dataset_tag=self.config.data.dataloader.test_dataset_tag,
            batch_size=self.config.training.test_batch_size,
            sampler=test_sampler,
            **self.config.data.dataloader.cuda_kwargs.model_dump(),
            collate_fn=collate_fn,
        )

        return data_loader_splits


if __name__ == "__main__":
    main()
