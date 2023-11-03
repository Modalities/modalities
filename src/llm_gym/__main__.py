import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import click_pathlib
import hydra
import torch.distributed as dist
import torch.optim as optim
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler

from .batch import EvaluationResultBatch
from .checkpointing.checkpointing import Checkpointing
from .checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from .checkpointing.checkpointing_strategies import SaveMostRecentEpochOnlyCheckpointingStrategy
from .config.config import AppConfig
from .dataloader.dataset import Dataset, MemMapDataset
from .dataset_loader import LLMDataLoader
from .evaluator import Evaluator
from .fsdp.fsdp_runner import Runner
from .gpt2.collator import GPT2LLMCollator
from .gpt2.gpt2_model import GPT2LLM
from .gym import Gym
from .logging_broker.message_broker import MessageBroker
from .logging_broker.messages import BatchProgressUpdate, MessageTypes
from .logging_broker.publisher import MessagePublisher
from .logging_broker.subscriber_impl.batch_progress_subscriber import DummyProgressSubscriber, RichProgressSubscriber
from .logging_broker.subscriber_impl.results_subscriber import WandBEvaluationResultSubscriber
from .loss_functions import CLMCrossEntropyLoss, Loss
from .trainer import Trainer
from .util import get_date_of_run


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
        dist_launched = dist.is_torchelastic_launched()
        self.config = config

        self.experiment_id = get_date_of_run()

        self.dataset_path = config.data.dataset_dir_path

        self.model: GPT2LLM = init_by_hydra(config.model)
        runner: Runner = init_by_hydra(config.runner)

        self.wrapped_model = runner.wrap(model=self.model, local_rank=config.globals.local_rank)
        self.optimizer = optim.AdamW(self.wrapped_model.parameters(), lr=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

        # CLMCrossEntropyLoss(target_subscription_key="target_key", prediction_subscription_key="logits")
        self.loss_fun: Loss = init_by_hydra(config.loss)

        # data loaders
        (
            self.train_dataloader,
            self.val_dataloader_1,
            self.val_dataloader_2,
        ), self.sampler_train = self.create_dataloaders(
            train_batch_size=config.globals.training_batch_size, test_batch_size=21
        )

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
            self.val_dataloader_1.dataset_tag: len(self.val_dataloader_1) * config.globals.world_size,
            self.val_dataloader_2.dataset_tag: len(self.val_dataloader_2) * config.globals.world_size,
        }
        train_split_lengths = {self.train_dataloader.dataset_tag: config.globals.num_training_batches}

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

        self.loss_fun = CLMCrossEntropyLoss(target_subscription_key="target_key", prediction_subscription_key="logits")

        # Checkpointing
        checkpointing_strategy = SaveMostRecentEpochOnlyCheckpointingStrategy()
        checkpointing_execution = FSDPToDiscCheckpointing(
            checkpoint_path="/raid/s3/opengptx/max_lue/LLMgym/checkpoints",
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
        self.eval_data_loaders = [self.val_dataloader_1, self.val_dataloader_2]

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
            num_batches=self.config.globals.num_batches_per_rank,
            num_batches_per_epoch=self.config.globals.num_batches_per_training_sequence_per_rank,
            train_data_loader=self.train_dataloader,
            evaluation_data_loaders=self.eval_data_loaders,
        )

    def create_dataloaders(
        self, train_batch_size: int, test_batch_size: int
    ) -> Tuple[List[LLMDataLoader], DistributedSampler]:
        # create dataset splits
        dataset_dict = Dataset.from_path(self.dataset_path, target_dataset_cls=MemMapDataset, split_size=(0.9, 0.1, 0))
        train_dataset = dataset_dict.train
        val_dataset = dataset_dict.validation

        # create samplers
        sampler_train = DistributedSampler(
            train_dataset,
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
            shuffle=True,
        )
        sampler_val_1 = DistributedSampler(
            val_dataset, rank=self.config.globals.global_rank, num_replicas=self.config.globals.world_size
        )
        sampler_val_2 = DistributedSampler(
            val_dataset, rank=self.config.globals.global_rank, num_replicas=self.config.globals.world_size
        )

        # create dataloaders
        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
        pad_to_multiple_of = 8
        collate_fn = GPT2LLMCollator(
            target_publication_key="target_key",
            pad_to_multiple_of=pad_to_multiple_of,
        )
        train_loader = LLMDataLoader(
            dataset=train_dataset,
            dataset_tag="train",
            batch_size=train_batch_size,
            sampler=sampler_train,
            **cuda_kwargs,
            collate_fn=collate_fn,
        )
        val_loader_1 = LLMDataLoader(
            dataset=val_dataset,
            dataset_tag="val_1",
            batch_size=test_batch_size,
            sampler=sampler_val_1,
            **cuda_kwargs,
            collate_fn=collate_fn,
        )
        val_loader_2 = LLMDataLoader(
            dataset=val_dataset,
            dataset_tag="val_2",
            batch_size=test_batch_size,
            sampler=sampler_val_2,
            **cuda_kwargs,
            collate_fn=collate_fn,
        )

        return [train_loader, val_loader_1, val_loader_2], sampler_train


if __name__ == "__main__":
    main()
