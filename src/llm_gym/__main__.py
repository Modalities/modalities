import logging
from pathlib import Path
from typing import Any, Callable, Dict

import click
import click_pathlib
import hydra
import numpy as np
import torch.distributed as dist
import torch.optim as optim
from datasets import Dataset
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch.optim.lr_scheduler import StepLR
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
from llm_gym.loss_functions import CLMCrossEntropyLoss, Loss
from llm_gym.models.gpt2.collator import GPT2LLMCollator
from llm_gym.models.gpt2.gpt2_model import GPT2LLM
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
    config_dict = hydra_load_app_config_dict(config_file_path)
    config = AppConfig.model_validate(config_dict)
    main = Main(config)
    main.run()


def hydra_load_app_config_dict(config_file_path: Path) -> Dict:
    hydra.initialize(config_path=config_file_path.parent.__str__())
    cfg = hydra.compose(config_name=config_file_path.stem.__str__())
    logging.info(f"Hydra\n {OmegaConf.to_yaml(cfg, resolve=True)}")
    return OmegaConf.to_container(cfg, resolve=True)


def init_by_hydra(config: BaseModel) -> Any:
    config_dict = config.model_dump()
    assert "target_class" in config_dict.keys()
    config_dict["_target_"] = config_dict.pop("target_class")
    return hydra.utils.instantiate(OmegaConf.create(config_dict))


class Main:
    def __init__(self, config: AppConfig) -> None:
        # Checks whether this process was launched with ``torch.distributed.elastic``
        dist_launched = dist.is_torchelastic_launched()
        self.config = config

        self.experiment_id = get_date_of_run()

        self.model: GPT2LLM = init_by_hydra(config.model)
        runner: Runner = init_by_hydra(config.runner)

        self.wrapped_model = runner.wrap(model=self.model, local_rank=config.globals.local_rank)
        self.optimizer = optim.AdamW(self.wrapped_model.parameters(), lr=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

        self.loss_fun: Loss = init_by_hydra(config.loss)

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
            train_batch_size=config.globals.training_batch_size,
            test_batch_size=config.globals.evaluation_batch_size,
            collate_fn=collator,
        )

        self.train_dataloader = dataloader_splits["train"]
        self.val_dataloader = dataloader_splits["val"]
        self.test_dataloader = dataloader_splits["test"]

        # Message Broker
        message_broker = MessageBroker()
        batch_processed_publisher = MessagePublisher[BatchProgressUpdate](
            message_broker=message_broker,
            global_rank=config.globals.global_rank,
            local_rank=config.globals.local_rank,
        )
        evaluation_result_publisher = MessagePublisher[EvaluationResultBatch](
            message_broker=message_broker,
            global_rank=config.globals.global_rank,
            local_rank=config.globals.local_rank,
        )

        eval_split_lengths = {
            self.val_dataloader.dataset_tag: len(self.val_dataloader) * config.globals.world_size,
            self.test_dataloader.dataset_tag: len(self.test_dataloader) * config.globals.world_size,
        }
        train_split_lengths = {self.train_dataloader.dataset_tag: len(self.train_dataloader)}

        if not dist_launched or (dist_launched and dist.get_rank() == 0):
            progress_subscriber = RichProgressSubscriber(
                num_ranks=config.globals.world_size,
                train_split_lengths=train_split_lengths,
                eval_split_lengths=eval_split_lengths,
            )
            evaluation_result_subscriber = WandBEvaluationResultSubscriber(
                num_ranks=config.globals.world_size, project="llm_gym", experiment_id=self.experiment_id
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

        self.loss_fun = CLMCrossEntropyLoss(
            target_key=config.loss.target_key,
            prediction_key=config.loss.prediction_key,
        )

        # Checkpointing
        checkpointing_strategy = SaveMostRecentEpochOnlyCheckpointingStrategy()
        checkpointing_execution = FSDPToDiscCheckpointing(
            checkpoint_path="/raid/s3/opengptx/mehdi/temp/temp_data",
            experiment_id=self.experiment_id,
            global_rank=config.globals.global_rank,
            checkpointing_rank=0,
        )
        checkpointing = Checkpointing(
            checkpointing_execution=checkpointing_execution,
            checkpointing_strategy=checkpointing_strategy,
            num_ranks=config.globals.world_size,
        )

        # Trainer
        self.trainer = Trainer(
            local_rank=config.globals.local_rank,
            batch_progress_publisher=batch_processed_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Evaluator
        self.eval_data_loaders = [self.val_dataloader, self.test_dataloader]

        self.evaluator = Evaluator(
            local_rank=config.globals.local_rank,
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
            # TODO: remove num_batches dependency
            num_batches_per_rank=self.config.globals.num_training_batches // self.config.globals.world_size,
            eval_interval_in_batches=self.config.globals.eval_interval_per_rank,
            train_data_loader=self.train_dataloader,
            evaluation_data_loaders=self.eval_data_loaders,
        )

    def create_instances(self, config: AppConfig) -> Dict[str, TextInstances]:
        # on the fly tokenization
        # from llm_gym.dataloader.dataset import Dataset as Dataset_Wrapper
        # from llm_gym.dataloader.dataset import MemMapDataset
        # dataset_dict = Dataset_Wrapper.from_path(config.data.dataset_dir_path, target_dataset_cls=MemMapDataset, split_size=(0.8, 0.1, 0.1)) # noqa: E501
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
            num_samples = config.globals.num_training_batches * config.globals.training_batch_size
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
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
            shuffle=True,
        )

        sampler_splits["val"] = DistributedSampler(
            dataset=val_instances,
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
        )

        sampler_splits["test"] = DistributedSampler(
            dataset=test_instances,
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
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
        train_batch_size: int,
        test_batch_size: int,
        collate_fn: Callable,
    ) -> Dict[str, LLMDataLoader]:
        """Create dataset splits."""

        # create dataloaders
        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
        data_loader_splits = dict()

        data_loader_splits["train"] = LLMDataLoader(
            dataset=train_instances,
            dataset_tag="train",
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            **cuda_kwargs,
        )

        data_loader_splits["val"] = LLMDataLoader(
            dataset=val_instances,
            dataset_tag="val",
            batch_size=test_batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            **cuda_kwargs,
        )

        data_loader_splits["test"] = LLMDataLoader(
            dataset=test_instances,
            dataset_tag="test",
            batch_size=test_batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn,
            **cuda_kwargs,
        )

        return data_loader_splits


if __name__ == "__main__":
    main()
