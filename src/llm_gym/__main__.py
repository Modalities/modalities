import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import click_pathlib
import hydra
from llm_gym.config.config import AppConfig
from omegaconf import OmegaConf
from pydantic import BaseModel


from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import (
    SaveMostRecentEpochOnlyCheckpointingStrategy,
)
from llm_gym.data.instances import TextInstances
from llm_gym.data.mmap_dataset import make_dataset
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.fsdp.fsdp_runner import FSDPRunner, Runner
from llm_gym.models.gpt2.gpt2_model import GPT2LLM
from llm_gym.models.gpt2.collator import GPT2LLMCollator, LMWikiBookCorpusDatasetFactory
from llm_gym.gym import Gym
from llm_gym.logging_broker.subscriber_impl.batch_progress_subscriber import (
    DummyProgressSubscriber,
    RichProgressSubscriber,
)
from llm_gym.logging_broker.subscriber_impl.results_subscriber import (
    WandBEvaluationResultSubscriber,
)
from llm_gym.trainer import Trainer
from llm_gym.evaluator import Evaluator
from llm_gym.loss_functions import CLMCrossEntropyLoss, Loss
from llm_gym.util import get_date_of_run
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from llm_gym.logging_broker.message_broker import MessageBroker
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.logging_broker.messages import BatchProgressUpdate
from llm_gym.logging_broker.messages import MessageTypes


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

        # Create data loaders
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
        self,
        config: Dict[str, Any],
        train_batch_size: int,
        test_batch_size: int,

    ) -> Tuple[List[LLMDataLoader], DistributedSampler]:
        """Create dataset splits."""
        dataset_path = config.data.dataset_dir_path
        sequence_len = config.data.sequence_len
        instance_splits = dict()

        for partition in ["train", "val", "test"]:
            dataset_filename_prefix = list(
                set([dataset_path.joinpath(filename.stem) for filename in dataset_path.glob(f"*{partition}*.bin")])
            )[0]
            text_dataset = make_dataset(path=self.dataset_path)
            instances = TextInstances(
                text_dataset=text_dataset,
                dataset_dir=self.dataset_path,
                num_samples=len(text_dataset),
                dataset_name=dataset_filename_prefix,
                sequence_len=sequence_len,
            )
            instance_splits[partition] = instances

        # create samplers
        sampler_train = DistributedSampler(
            dataset=instance_splits["train"],
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
            shuffle=True,
        )
        sampler_val = DistributedSampler(
            dataset=instance_splits["val"],
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
        )
        sampler_test = DistributedSampler(
            dataset=instance_splits["test"],
            rank=self.config.globals.global_rank,
            num_replicas=self.config.globals.world_size,
        )

        # create dataloaders
        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}

        train_loader = LLMDataLoader(
            dataset=instances["train"],
            dataset_tag="train",
            batch_size=train_batch_size,
            sampler=sampler_train,
            **cuda_kwargs,
            #collate_fn=collate_fn,
        )
        val_loader = LLMDataLoader(
            dataset=instances["val"],
            dataset_tag="val_1",
            batch_size=test_batch_size,
            sampler=sampler_val,
            **cuda_kwargs,
            #collate_fn=collate_fn,
        )
        test_loader = LLMDataLoader(
            dataset=instances["test"],
            dataset_tag="val_2",
            batch_size=test_batch_size,
            sampler=sampler_test,
            **cuda_kwargs,
            #collate_fn=collate_fn,
        )

        return [train_loader, val_loader, test_loader], sampler_train


if __name__ == "__main__":
    main()
