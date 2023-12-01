import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import click
import click_pathlib
from llm_gym.config.config import AppConfig, DataLoaderConfig, DatasetConfig, LLMDataLoaderConfig, SamplerConfig, DataLoaderConfig
import numpy as np
from pydantic import DirectoryPath
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import (
    SaveAllCheckpointingStrategy,
    SaveMostRecentEpochOnlyCheckpointingStrategy
)
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.fsdp.fsdp_running_env import RunningEnv
from llm_gym.models.gpt2.collator import GPT2LLMCollator

from llm_gym.data.instances import TextInstances
from llm_gym.data.mmap_dataset import make_dataset
from llm_gym.dataset_loader import LLMDataLoader


import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from llm_gym.batch import EvaluationResultBatch
from llm_gym.dataloader.create_index import create_memmap_index
from llm_gym.dataloader.create_packed_data import create_packed_data
from llm_gym.evaluator import Evaluator
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
import torch.nn as nn
from torch.optim import Optimizer
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
        self.config = config

        # warmstart
        self.global_train_batch_id = 139999
        self.warmstart_experiment_id = "2023-11-16-07:42:45_PM"

        # coldstart
        self.experiment_id = get_date_of_run()

        self.resolvers = ResolverRegister(config=config)
        self.running_env: RunningEnv = self.resolvers.build_component_by_config(config=self.config.running_env)


    def construct_components(
        self, resolvers: ResolverRegister, config: AppConfig, running_env: RunningEnv
    ) -> Tuple[Gym, LLMDataLoader, List[LLMDataLoader], Checkpointing, nn.Module, Optimizer]:
  
        # Checkpointing
        checkpointing = self.get_checkpointing(config=config, running_env=running_env)

        # Model and optimizer
        wrapped_model, optimizer = self.get_model_and_optimizer(
            config=config, running_env=running_env, checkpointing=checkpointing
        )
        # Loss fun
        loss_fun: Loss = resolvers.build_component_by_config(config=config.loss)

        # Dataloaders
        # TODO  Max's version
        train_dataloader, val_dataloader, test_dataloader = self.get_dataloaders(config=config)


        # TODO Lucian's and Viktor's version
        # train_dataloader = self._create_dataloader(config=config.training.train_dataloader)
        # validation_dataloader_lookup = {
        #     dataset_tag: self._create_dataloader(config=config)
        #     for dataset_tag, config in config.training.evaluation_dataloaders.items()
        # }
        # val_dataloader = validation_dataloader_lookup["val"]
        # test_dataloader = validation_dataloader_lookup["test"]

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
            ) = self.construct_components(resolvers=self.resolvers, config=self.config, running_env=running_env)

            gym.run(
                num_training_batches_per_rank=self.config.training.num_training_batches_per_rank,
                eval_interval_per_rank=self.config.training.eval_interval_per_rank,
                train_data_loader=train_dataloader,
                evaluation_data_loaders=eval_data_loaders,
                checkpointing=checkpointing,
                model=wrapped_model,
                optimizer=optimizer,
            )


    def get_dataloaders(self, config: AppConfig) -> List[LLMDataLoader]:
        train_dataloader_config = config.data.train_dataloader
        eval_dataloader_configs = config.data.eval_dataloaders
        data_loaders = []

        # Dataloaders
        collator = GPT2LLMCollator(
            sample_key=config.data.sample_key,
            target_key=config.data.target_key,
        )

        for dataloader_config in [train_dataloader_config, *eval_dataloader_configs]:
            # Create instances
            split_instances = self.create_instances(
                dataset_config=dataloader_config.config.dataset,  # TODO dataset config should not be passed like this
            )

            # Create samplers
            # TODO this should be instantiated automatically in the registry by traversing the dependency tree
            split_sampler = self.create_sampler(
                sampler_config=dataloader_config.config.sampler, instances=split_instances
            )

            # create dataloaders
            dataloader_split = self.create_dataloader(
                dataloader_config=dataloader_config,
                instances=split_instances,
                sampler=split_sampler,
                collate_fn=collator,
            )
            data_loaders.append(dataloader_split)
        return data_loaders

    def get_model_and_optimizer(
        self, config: AppConfig, running_env: RunningEnv, checkpointing: Checkpointing
    ) -> Tuple[nn.Module, Optimizer]:

        # self.model: torch.nn.Module = self.resolvers.build_component_by_config(config=config.model)

        # runner: Runner = self.resolvers.build_component_by_config(config=config.runner)

        # self.wrapped_model = runner.wrap(model=self.model, local_rank=config.training.local_rank)

        # self.optimizer: torch.optim.Optimizer = self.resolvers.build_component_by_config(
        #     config=config.optimizer, extra_kwargs=dict(params=self.wrapped_model.parameters())
        # )

        # self.scheduler = self.resolvers.build_component_by_config(
        #     config=config.scheduler, extra_kwargs=dict(optimizer=self.optimizer)
        # )

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
        dataset_config: DatasetConfig,
    ) -> TextInstances:
        dataset_directory = dataset_config.path.parents[0]
        dataset_filename_prefix = dataset_directory.joinpath(dataset_config.path.stem)
        text_dataset = make_dataset(path=dataset_filename_prefix)
        instances = TextInstances(
            sample_key=dataset_config.sample_key,
            text_dataset=text_dataset,
            doc_idx=np.arange(0, len(text_dataset)),
            dataset_dir=dataset_directory,
            num_samples=dataset_config.num_samples,
            dataset_name=dataset_config.dataset_tag,
            sequence_len=dataset_config.sequence_len,
        )
        return instances

    def create_sampler(self, sampler_config: SamplerConfig, instances: TextInstances) -> DistributedSampler:
        sampler: Sampler = self.resolvers.build_component_by_config(
            config=sampler_config, extra_kwargs=dict(dataset=instances)
        )
        return sampler

    def create_dataloader(
        self,
        dataloader_config: DataLoaderConfig,
        instances: Dataset,
        sampler: DistributedSampler,
        collate_fn: Callable,
        **dataloader_kwargs,
    ) -> LLMDataLoader:
        collate_fn = GPT2LLMCollator(
            sample_key=self.config.data.sample_key,
            target_key=self.config.data.target_key,
        )

        data_loader: LLMDataLoader = self.resolvers.build_component_by_config(
            config=dataloader_config, extra_kwargs=dict(dataset=instances, sampler=sampler, collate_fn=collate_fn)
        )
        # data_loader = LLMDataLoader(
        #     dataset=instances,
        #     dataset_tag=dataset_tag,
        #     batch_size=batch_size,
        #     sampler=sampler,
        #     collate_fn=collate_fn,
        #     **dataloader_kwargs,
        # )
        return data_loader

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


        return evaluation_result_publisher, batch_processed_publisher

    # TODO use this as the default factory for dataloaders 
    def _create_dataloader(self, config: DataLoaderConfig) -> DataLoader:
        dataset = self.resolvers.build_component_by_config(config=config.config.dataset)
        collator = self.resolvers.build_component_by_config(config=config.config.collate_fn)
        sampler = self.resolvers.build_component_by_config(
            config=config.config.sampler, extra_kwargs=dict(dataset=dataset)
        )
        return self.resolvers.build_component_by_config(
            config=config,
            extra_kwargs=dict(
                dataset=dataset,
                sampler=sampler,
                collate_fn=collator,
            ),
        )


if __name__ == "__main__":
    main()
