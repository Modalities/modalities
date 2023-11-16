import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import click
import click_pathlib
from llm_gym.config.config import AppConfig
from llm_gym.logging_broker.subscriber import MessageSubscriberIF
from llm_gym.optimizers.optimizer_factory import OptimizerFactory, OptimizerTypes
import numpy as np
from pydantic import DirectoryPath
import torch
import torch.distributed as dist
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import (
    SaveAllCheckpointingStrategy,
)
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.fsdp.fsdp_running_env import FSDPRunningEnv, RunningEnv
from llm_gym.models.gpt2.gpt2_model import GPT2LLM
from llm_gym.models.gpt2.collator import GPT2LLMCollator

from llm_gym.checkpointing.checkpointing_strategies import SaveMostRecentEpochOnlyCheckpointingStrategy
from llm_gym.config.config import AppConfig
from llm_gym.data.instances import TextInstances
from llm_gym.data.mmap_dataset import make_dataset
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.fsdp.fsdp_running_env import RunningEnv, FSDPRunningEnv
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
import torch.nn as nn
from torch.optim import Optimizer


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
        self.config = config

        # warmstart
        self.global_train_batch_id = 0
        self.warmstart_experiment_id = "2023-11-15-11:53:54_PM"

        self.experiment_id = get_date_of_run()
        self.dataset_path = config.data.dataset_dir_path

        self.resolvers = ResolverRegister(config=config)
        self.running_env: RunningEnv = self.resolvers.build_component_by_config(config=self.config.running_env)

    def construct_components(
        self, config: AppConfig, running_env: RunningEnv
    ) -> Tuple[Gym, LLMDataLoader, List[LLMDataLoader], Checkpointing, nn.Module, Optimizer]:
        # TODO move to run function, despite mixing concerns here ...
        # self.wrapped_model = running_env.wrap(model=self.model, local_rank=config.training.local_rank)

        # self.optimizer: torch.optim.Optimizer = resolvers.build_component_by_config(
        #     config=config.optimizer, extra_kwargs=dict(params=self.wrapped_model.parameters())
        # )

        # self.scheduler = resolvers.build_component_by_config(
        #     config=config.scheduler, extra_kwargs=dict(optimizer=self.optimizer)
        # )

        # Dataloaders
        collator = GPT2LLMCollator(
            sample_key=config.data.sample_key,
            target_key=config.data.target_key,
        )
        train_dataloader, val_dataloader, test_dataloader = self.get_dataloaders(config=config, collator=collator)

        # Checkpointing
        checkpointing = self.get_checkpointing(config=config, running_env=running_env)

        wrapped_model, optimizer = self.get_model_and_optimizer(
            config=config, running_env=running_env, checkpointing=checkpointing
        )

        # Loss function
        loss_fun: Loss = self.resolvers.build_component_by_config(config=config.loss)



        # Logging
        eval_split_lengths = {
            val_dataloader.dataset_tag: len(val_dataloader) * config.training.world_size,
            test_dataloader.dataset_tag: len(test_dataloader) * config.training.world_size,
        }
        train_split_lengths = {train_dataloader.dataset_tag: len(train_dataloader)}

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

    def run(self):
        with self.running_env as running_env:
            (
                gym,
                train_dataloader,
                eval_data_loaders,
                checkpointing,
                wrapped_model,
                optimizer,
            ) = self.construct_components(self.config, running_env=running_env)

            gym.run(
                num_batches_per_rank=self.config.training.num_batches_per_rank,
                eval_interval_in_batches=self.config.training.eval_interval_in_batches,
                train_data_loader=train_dataloader,
                evaluation_data_loaders=eval_data_loaders,
                checkpointing=checkpointing,
                model=wrapped_model,
                optimizer=optimizer,
            )

    def get_dataloaders(self, config: AppConfig, collator: Callable) -> Tuple[LLMDataLoader, LLMDataLoader, LLMDataLoader]:
        # Create instances
        instance_splits = self.create_instances(
            dataset_dir_path=config.data.dataset_dir_path,
            sequence_len=config.data.sequence_len,
            num_training_batches=config.training.num_training_batches,
            training_batch_size=config.training.training_batch_size,
            sample_key=config.data.sample_key,
        )

        # Create samplers
        sampler_splits = self.create_samplers(
            train_instances=instance_splits["train"],
            val_instances=instance_splits["val"],
            test_instances=instance_splits["test"],
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
        return dataloader_splits["train"], dataloader_splits["val"], dataloader_splits["test"]

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

        # lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # TODO use lr_scheduler

        return wrapped_model, optimizer

    def get_checkpointing(self, config: AppConfig, running_env: RunningEnv) -> Checkpointing:
        checkpointing_strategy = SaveAllCheckpointingStrategy()
        checkpointing_execution = FSDPToDiscCheckpointing(
            checkpoint_path="/raid/s3/opengptx/max_lue/LLMgym/checkpoints",
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

    def create_instances(
        self,
        dataset_dir_path: DirectoryPath,
        sequence_len: int,
        num_training_batches: int,
        training_batch_size: int,
        sample_key: str,
    ) -> Dict[str, TextInstances]:
        instance_splits = dict()
        for partition in ["train", "val", "test"]:
            dataset_filename_prefix = list(
                set(
                    [
                        dataset_dir_path.joinpath(filename.stem)
                        for filename in dataset_dir_path.glob(f"*{partition}*.bin")
                    ]
                )
            )[0]
            text_dataset = make_dataset(path=dataset_filename_prefix)
            num_samples = num_training_batches * training_batch_size
            instances = TextInstances(
                sample_key=sample_key,
                text_dataset=text_dataset,
                doc_idx=np.arange(0, len(text_dataset)),
                dataset_dir=dataset_dir_path,
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
