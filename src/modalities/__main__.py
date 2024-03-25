#!/usr/bin/env python

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import click
import click_pathlib

from modalities.activation_checkpointing import apply_activation_checkpointing_inplace
from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ComponentsModel, ProcessGroupBackendType, TokenizerTypes, load_app_config_dict
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import BatchProgressUpdate, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.trainer import Trainer
from modalities.util import compute_number_of_trainable_parameters, get_callback_interval_in_batches_per_rank
from modalities.utils.generate_text import main as generate_text_main
from modalities.utils.gradient_clipping import build_gradient_clipper


@click.group()
def main() -> None:
    pass


config_file_path_option = click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)


@main.command(name="run")
@config_file_path_option
def entry_point_run_modalities(config_file_path: Path):
    config_dict = load_app_config_dict(config_file_path)
    main = Main(config_dict, config_file_path)
    main.run()


@main.command(name="generate_text")
@click.argument("model_path", type=Path)
@config_file_path_option
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
    default=Path(__file__).resolve().parents[2] / Path("data/tokenizer/tokenizer.json"),
    help="path to tokenizer json",
)
@click.option("--max_new_tokens", type=int, show_default=True, default=200, help="maximum amount of tokens to generate")
@click.option("--chat", is_flag=True, show_default=True, default=False, help="activate 'chat' mode")
def entry_point_generate_text(model_path, config_file_path, tokenizer_type, tokenizer_file, max_new_tokens, chat):
    tokenizer = tokenizer_type.value(tokenizer_file=str(tokenizer_file))
    generate_text_main(model_path, config_file_path, tokenizer, max_new_tokens, chat)


@main.group(name="data")
def data():
    """
    Collection of utilities to preprocess, analyse and modify training data.
    """
    pass


@data.command(name="create_raw_index")
@click.argument("src_path", type=Path)
@click.option(
    "--index_path",
    type=Path,
    default=None,
    help="output path for index. will use parent directory of src_path if none.",
)
def entry_point_data_create_raw_index(src_path, index_path):
    """
    Utility for indexing a large jsonl-file's content.
    Background is the ability to further process the respective file without loading it,
    while splitting its content line-based. This step is necessary in advance of further processing like tokenization.
    It is only necessary once for a jsonl-file and allows therefore different tokenizations without re-indexing.
    """
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        raise ValueError("index already exists. delete it or specify different output folder.")

    print(f"reading raw data from {src_path}")
    print(f"writing index to {index_path}")
    generator = IndexGenerator(src_path)
    generator.create_index(index_path)


@data.command(name="pack_encoded_data")
@click.argument("src_path", type=Path)
@click.option(
    "--dst_path",
    type=str,
    default=None,
    help="output path for packed data file. will use parent directory of src_path if none.",
)
@click.option(
    "--index_path",
    type=Path,
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
    default=Path(__file__).resolve().parents[2] / Path("data/tokenizer/tokenizer.json"),
    help="path to tokenizer json",
)
@click.option(
    "--jq_pattern",
    type=str,
    show_default=True,
    default=".text",
    help="jq pattern to extract the data from the json line.",
)
@click.option(
    "--num-cpus",
    type=int,
    show_default=True,
    default=os.cpu_count(),
    help="Specify the number of tokenization workers. Default is the number of available CPUs.",
)
def entry_point_pack_encoded_data(src_path, dst_path, index_path, tokenizer_type, tokenizer_file, jq_pattern, num_cpus):
    """
    Utility to encode an indexed, large jsonl-file.

    (see also `create_index` for more information)
    Returns .pbin-file, which can be inserted into a training process directly
    and does not require its original jsonl-file or the respective index file anymore.
    """
    # TODO: if we want to use alternative entrypoints together with the ResolverRegistry,
    #  we can currently not rely on the existing class resolver.
    #  This is based on its connection to the overall `AppConfig`.
    #  One would requires an object of it to instantiate the ResolverRegistry.
    #  This could get resolved by implementing on own ResolverRegistry for each entrypoint or adapting the existing
    #  ResolverRegistry to work dynamically with any type-hinted config object from config.py.
    tokenizer = tokenizer_type.value(tokenizer_file=str(tokenizer_file))
    generator = PackedDataGenerator(
        src_path,
        index_path=index_path,
        tokenizer=tokenizer,
        jq_pattern=jq_pattern,
        number_of_processes=num_cpus,
    )
    generator.run(dst_path)


@main.command(name="convert_pytorch_to_hf_checkpoint")
@click.option(
    "--checkpoint_dir",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Load pytorch checkpoint from this directory.",
)
@click.option(
    "--config_file_name",
    type=str,
    required=False,
    default="model_config.yaml",
    help="Name of the config file for the input pytorch checkpoint, which must be located in checkpoint_dir.",
)
@click.option(
    "--model_file_name",
    type=str,
    required=False,
    default="model.bin",
    help="Name of the model file for the input pytorch checkpoint, which must be located in checkpoint_dir.",
)
@click.option(
    "--output_hf_checkpoint_dir",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Converted hf checkpoint will be written to this directory.",
)
def entry_point_convert_pytorch_to_hf_checkpoint(
    checkpoint_dir: Path, config_file_name: str, model_file_name: str, output_hf_checkpoint_dir: Path
):
    cp = CheckpointConversion(checkpoint_dir, config_file_name, model_file_name, output_hf_checkpoint_dir)
    cp.convert_pytorch_to_hf_checkpoint()


@data.command(name="merge_packed_data")
@click.argument("src_paths", type=click.types.Path(exists=True, path_type=Path), nargs=-1, required=True)
@click.argument("target_path", type=click.types.Path(file_okay=False, dir_okay=False, path_type=Path))
def entry_point_merge_packed_data(src_paths, target_path):
    """
    Utility for merging different pbin-files into one.
    This is especially useful, if different datasets were at different points in time or if one encoding takes so long,
    that the overall process was done in chunks.
    It is important that the same tokenizer got used for all chunks.

    Specify an arbitrary amount of pbin-files and/or directory containing such as input.
    """
    input_files = []
    for p in src_paths:
        p: Path
        if p.is_dir():
            input_files.extend(p.glob("**/*.pbin"))
        else:
            input_files.append(p)
    embedded_datasets = list(map(EmbeddedStreamData, input_files))
    join_embedded_stream_data(embedded_datasets, target_path)


class Main:
    def __init__(self, config_dict: Dict, config_path: Path) -> None:
        self.config_dict = config_dict
        self.config_path = config_path

        self.registry = Registry(COMPONENTS)
        self.component_factory = ComponentFactory(registry=self.registry)

    def add_custom_component(self, component_key: str, variant_key: str, custom_component, custom_config) -> None:
        self.registry.add_entity(
            component_key=component_key,
            variant_key=variant_key,
            component_type=custom_component,
            component_config_type=custom_config,
        )

    def run(self):
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            components: ComponentsModel = self.component_factory.build_components(
                config_dict=self.config_dict, components_model_type=ComponentsModel
            )

            # save the config file to the checkpointing path
            if components.settings.cuda_env.global_rank == 0:
                experiment_path = components.settings.paths.checkpointing_path / components.settings.experiment_id
                os.makedirs(experiment_path, exist_ok=True)
                shutil.copy(self.config_path, experiment_path / self.config_path.name)

            evaluation_result_publisher, batch_processed_publisher = self.get_logging_publishers(
                progress_subscriber=components.batch_progress_subscriber,
                results_subscriber=components.evaluation_subscriber,
                global_rank=components.settings.cuda_env.global_rank,
                local_rank=components.settings.cuda_env.local_rank,
            )

            # Trainer
            trainer = Trainer(
                local_rank=components.settings.cuda_env.local_rank,
                batch_progress_publisher=batch_processed_publisher,
                evaluation_result_publisher=evaluation_result_publisher,
                gradient_acc_steps=components.settings.training.gradient_acc_steps,
                gradient_clipper=build_gradient_clipper(
                    gradient_clipping_mode=components.settings.training.gradient_clipping.mode,
                    gradient_clipping_threshold=components.settings.training.gradient_clipping.threshold,
                ),
            )

            # Evaluator
            evaluator = Evaluator(
                local_rank=components.settings.cuda_env.local_rank,
                batch_progress_publisher=batch_processed_publisher,
                evaluation_result_publisher=evaluation_result_publisher,
            )

            # Gym
            gym = Gym(
                trainer=trainer,
                evaluator=evaluator,
                loss_fun=components.loss_fn,
                num_ranks=components.settings.cuda_env.world_size,
            )
            wrapped_model = components.wrapped_model
            logging.info(f"Training model with {compute_number_of_trainable_parameters(wrapped_model)} parameters.")

            if components.settings.training.do_apply_activation_checkpointing:
                apply_activation_checkpointing_inplace(wrapped_model)

            callback_interval_in_batches_per_rank = get_callback_interval_in_batches_per_rank(
                callback_interval_in_samples=components.settings.training.callback_interval_in_samples,
                local_train_micro_batch_size=components.settings.training.local_train_micro_batch_size,
                gradient_acc_steps=components.settings.training.gradient_acc_steps,
                world_size=components.settings.cuda_env.world_size,
            )

            gym.run(
                callback_interval_in_batches=callback_interval_in_batches_per_rank,
                train_data_loader=components.train_dataloader,
                evaluation_data_loaders=components.eval_dataloaders,
                checkpointing=components.checkpointing,
                model=wrapped_model,
                optimizer=components.optimizer,
                scheduler=components.scheduler,
            )
            print("done")

    def get_logging_publishers(
        self,
        progress_subscriber: MessageSubscriberIF[BatchProgressUpdate],
        results_subscriber: MessageSubscriberIF[EvaluationResultBatch],
        global_rank: int,
        local_rank: int,
    ) -> Tuple[
        MessagePublisher[EvaluationResultBatch],
        MessagePublisher[BatchProgressUpdate],
    ]:
        message_broker = MessageBroker()
        batch_processed_publisher = MessagePublisher[BatchProgressUpdate](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        evaluation_result_publisher = MessagePublisher[EvaluationResultBatch](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )

        message_broker.add_subscriber(subscription=MessageTypes.EVALUATION_RESULT, subscriber=results_subscriber)
        message_broker.add_subscriber(
            subscription=MessageTypes.BATCH_PROGRESS_UPDATE,
            subscriber=progress_subscriber,
        )

        return evaluation_result_publisher, batch_processed_publisher


if __name__ == "__main__":
    main()
