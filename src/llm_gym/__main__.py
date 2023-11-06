import logging
import os
from pathlib import Path
from typing import Any, Dict

import click
import click_pathlib
import hydra
import torch.distributed as dist
import torch.optim as optim
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler


from llm_gym.callbacks.batch_progress_callbacks import DummyProgressCallback, PrintProgressCallback
from llm_gym.callbacks.results_callbacks import DummyResultsCallback, ResultsCallback
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from llm_gym.checkpointing.checkpointing_strategies import SaveMostRecentEpochOnlyCheckpointingStrategy
from llm_gym.config.config import AppConfig
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.forward_pass import ModelInferenceComponent
from llm_gym.fsdp.fsdp_runner import Runner
from llm_gym.gpt2.collator import GPT2LLMCollator, LMWikiBookCorpusDatasetFactory
from llm_gym.gpt2.gpt2_model import GPT2LLM, GPTConfig
from llm_gym.gym import Gym
from llm_gym.loss_functions import Loss
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
        self.experiment_id = get_date_of_run()

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.dataset_path = config.data.dataset_dir_path
        self.num_epochs = config.training.num_epochs

        self.model: GPT2LLM = init_by_hydra(config.model)
        runner: Runner = init_by_hydra(config.runner)
        self.wrapped_model = runner.wrap(model=self.model, local_rank=self.local_rank)
        self.model_inference_component = ModelInferenceComponent(model=self.wrapped_model, post_processors=[])

        self.optimizer = optim.AdamW(self.wrapped_model.parameters(), lr=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

        # CLMCrossEntropyLoss(target_subscription_key="target_key", prediction_subscription_key="logits")
        self.loss_fun: Loss = init_by_hydra(config.loss)

        # data loaders
        self.data_loaders, self.sampler_train = self.create_dataloaders(
            train_batch_size=32, test_batch_size=32
        )  # TODO make dynamic

        # Trainer
        train_split_key = "train"
        self.train_data_loader = self.data_loaders[train_split_key]
        train_split_length = {train_split_key: len(self.train_data_loader)}

        if dist.get_rank() == 0:
            # train_batch_processed_callback = RichProgressCallback(
            #   subscribing_global_rank=self.global_rank,
            #   num_epochs=self.num_epochs, split_lengths=train_split_length)
            train_batch_processed_callback = PrintProgressCallback(
                subscribing_global_rank=self.global_rank,
                num_epochs=self.num_epochs,
                split_lengths=train_split_length,
                print_frequency=0.1,
            )
            train_results_callback = ResultsCallback(subscribing_global_rank=self.global_rank)

        else:
            train_batch_processed_callback = DummyProgressCallback()
            train_results_callback = DummyResultsCallback()

        # Checkpointing
        checkpointing_strategy = SaveMostRecentEpochOnlyCheckpointingStrategy()
        checkpointing_execution = FSDPToDiscCheckpointing(
            checkpoint_path="checkpoints",
            experiment_id=self.experiment_id,
            global_rank=self.global_rank,
            checkpointing_rank=0,
        )
        checkpointing = Checkpointing(
            checkpointing_execution=checkpointing_execution, checkpointing_strategy=checkpointing_strategy
        )

        # Trainer
        self.trainer = Trainer(
            local_rank=self.local_rank,
            batch_processed_callback=train_batch_processed_callback,
            results_callback=train_results_callback,
        )

        # Evaluator
        val_split_key = "val"
        val_data_loader = self.data_loaders[val_split_key]
        self.eval_data_loaders = [val_data_loader]
        eval_split_lengths = {val_split_key: len(val_data_loader)}

        if dist.get_rank() == 0:
            # eval_batch_processed_callback = RichProgressCallback(
            #   subscribing_global_rank=self.global_rank,
            #   num_epochs=self.num_epochs, split_lengths=eval_split_lengths)
            eval_batch_processed_callback = PrintProgressCallback(
                subscribing_global_rank=self.global_rank,
                num_epochs=self.num_epochs,
                split_lengths=eval_split_lengths,
                print_frequency=0.1,
            )
            eval_results_callback = ResultsCallback(subscribing_global_rank=self.global_rank)
        else:
            eval_batch_processed_callback = DummyProgressCallback()
            eval_results_callback = DummyResultsCallback()

        self.evaluator = Evaluator(
            local_rank=self.local_rank,
            batch_processed_callback=eval_batch_processed_callback,
            results_callback=eval_results_callback,
        )

        # Gym
        self.gym = Gym(
            checkpointing=checkpointing,
            trainer=self.trainer,
            evaluator=self.evaluator,
            model_inference_component=self.model_inference_component,
            optimizer=self.optimizer,
            loss_fun=self.loss_fun,
        )

    def run(self):
        self.gym.run(
            num_epochs=self.num_epochs,
            train_data_loader=self.train_data_loader,
            evaluation_data_loaders=self.eval_data_loaders,
            sampler=self.sampler_train,
        )

    def create_dataloaders(self, train_batch_size: int, test_batch_size: int) -> Dict[str, LLMDataLoader]:
        # create dataset splits
        dataset_dict = LMWikiBookCorpusDatasetFactory.construct(self.dataset_path)
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]

        # create samplers
        sampler_train = DistributedSampler(
            train_dataset, rank=self.global_rank, num_replicas=self.world_size, shuffle=True
        )
        sampler_val = DistributedSampler(val_dataset, rank=self.global_rank, num_replicas=self.world_size)

        # create dataloaders
        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
        pad_to_multiple_of = 8
        tokenizer_file_path = "data/tokenizer/tokenizer.json"
        collate_fn = GPT2LLMCollator(
            target_publication_key="target_key",
            tokenizer_file_path=tokenizer_file_path,
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
        val_loader = LLMDataLoader(
            dataset=val_dataset,
            dataset_tag="val",
            batch_size=test_batch_size,
            sampler=sampler_val,
            **cuda_kwargs,
            collate_fn=collate_fn,
        )

        return {train_loader.dataset_tag: train_loader, val_loader.dataset_tag: val_loader}, sampler_train


if __name__ == "__main__":
    main()
