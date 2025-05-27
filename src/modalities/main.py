import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Type

import yaml
from pydantic import BaseModel

from modalities.batch import EvaluationResultBatch
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel, TrainingReportGenerator
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.trainer import Trainer
from modalities.util import get_synced_experiment_id_of_run, get_total_number_of_trainable_parameters, print_rank_0


class Main:
    """Main class that orchestrates the training process."""

    def __init__(
        self,
        config_path: Path,
        additional_resolver_funs: Optional[dict[str, Callable]] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        if experiment_id is None:
            experiment_id = get_synced_experiment_id_of_run(config_path)

        self.config_dict = load_app_config_dict(
            config_file_path=config_path, experiment_id=experiment_id, additional_resolver_funs=additional_resolver_funs
        )
        self.config_path = config_path

        self.registry = Registry(COMPONENTS)
        self.component_factory = ComponentFactory(registry=self.registry)

    def add_custom_component(
        self, component_key: str, variant_key: str, custom_component: Type, custom_config: Type
    ) -> None:
        """Add a custom component to the registry.

        This method comes in especially handy
        when Modalities is used as a library and the user wants to add custom components
        (e.g., custom model or custom loss function) to the registry.

        Args:
            component_key (str): Key of the component to be added to the registry
            variant_key (str): Key of the variant to be added to the registry
            custom_component (Type): The class type of the custom component
            custom_config (Type): The pydantic config type of the custom component
        """
        self.registry.add_entity(
            component_key=component_key,
            variant_key=variant_key,
            component_type=custom_component,
            component_config_type=custom_config,
        )

    def build_components(self, components_model_type: Type[BaseModel]) -> BaseModel:
        """Given a pydantic basemodel, this method builds the components specified in the config file.

        Depending on the use case (e.g., training, inference, etc.), the user can pass different pydantic base models.
        For instance, for tokenization, the basemodel would only have the tokenization-related components specified.

        Args:
            components_model_type (Type[BaseModel]): The pydantic basemodel type that should be
                used to build the components.

        Returns:
            BaseModel: The components built based on the config file.
        """
        components = self.component_factory.build_components(
            config_dict=self.config_dict, components_model_type=components_model_type
        )
        return components

    def run(self, components: TrainingComponentsInstantiationModel):
        """Entrypoint fo running the training process.

        We pass in a TrainingComponentsInstantiationModel,
        which is a pydantic model that contains all the components needed for the training process.

        Args:
            components (TrainingComponentsInstantiationModel): The components needed for the training process.
        """
        # save the config file to the checkpointing path
        if components.settings.cuda_env.global_rank == 0:
            experiment_path = components.settings.paths.checkpoint_saving_path / components.settings.experiment_id
            os.makedirs(experiment_path, exist_ok=True)
            shutil.copy(self.config_path, experiment_path / self.config_path.name)
            resolved_config_path = (experiment_path / self.config_path.name).with_suffix(".yaml.resolved")
            with open(resolved_config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config_dict, f)

        evaluation_result_publisher, progress_publisher = self.get_logging_publishers(
            progress_subscriber=components.progress_subscriber,
            results_subscriber=components.evaluation_subscriber,
            global_rank=components.settings.cuda_env.global_rank,
            local_rank=components.settings.cuda_env.local_rank,
        )

        # Trainer
        global_num_tokens_per_train_step = (
            components.settings.step_profile.local_train_micro_batch_size
            * components.settings.step_profile.sequence_length
            * components.settings.step_profile.gradient_accumulation_steps
            * components.settings.cuda_env.world_size
        )
        trainer = Trainer(
            global_rank=components.settings.cuda_env.global_rank,
            progress_publisher=progress_publisher,
            num_target_steps=components.settings.training_target.num_target_steps,
            num_target_tokens=components.settings.training_target.num_target_tokens,
            num_seen_train_steps=components.settings.training_progress.num_seen_steps,
            global_num_seen_tokens=components.settings.training_progress.global_num_seen_tokens,
            evaluation_result_publisher=evaluation_result_publisher,
            gradient_acc_steps=components.settings.step_profile.gradient_accumulation_steps,
            gradient_clipper=components.gradient_clipper,
            global_num_tokens_per_train_step=global_num_tokens_per_train_step,
            mfu_calculator=components.mfu_calculator,
        )

        # Evaluator
        evaluator = Evaluator(
            progress_publisher=progress_publisher,
            evaluation_result_publisher=evaluation_result_publisher,
        )

        # Gym
        gym = Gym(
            trainer=trainer,
            evaluator=evaluator,
            loss_fun=components.loss_fn,
            num_ranks=components.settings.cuda_env.world_size,
        )
        num_params = get_total_number_of_trainable_parameters(components.app_state.model)
        components.evaluation_subscriber.consume_dict({"No. parameters": num_params})
        logging.info(f"Training model with {num_params} parameters.")

        print_rank_0(f"Model initialized at {datetime.now()}.")

        report = TrainingReportGenerator(
            training_target=components.settings.training_target,
            intervals=components.settings.intervals,
            step_profile=components.settings.step_profile,
            cuda_env=components.settings.cuda_env,
            consistency_enforcement=components.settings.consistency_enforcement,
            train_dataset=components.train_dataset,
            training_progress=components.settings.training_progress,
        ).get_report()

        print_rank_0(report)

        gym.run(
            train_data_loader=components.train_dataloader,
            evaluation_data_loaders=components.eval_dataloaders,
            checkpoint_saving=components.checkpoint_saving,
            app_state=components.app_state,
            checkpointing_interval_in_steps=components.settings.intervals.checkpointing_interval_in_steps,
            evaluation_interval_in_steps=components.settings.intervals.evaluation_interval_in_steps,
            training_log_interval_in_steps=components.settings.intervals.training_log_interval_in_steps,
        )

    def get_logging_publishers(
        self,
        progress_subscriber: MessageSubscriberIF[ProgressUpdate],
        results_subscriber: MessageSubscriberIF[EvaluationResultBatch],
        global_rank: int,
        local_rank: int,
    ) -> tuple[MessagePublisher[EvaluationResultBatch], MessagePublisher[ProgressUpdate]]:
        """Returns the logging publishers for the training.

        These publishers are used to pass the evaluation results and the progress updates to the message broker.
        The message broker is then used to pass the messages to the subscribers, such as WandB.

        Args:
            progress_subscriber (MessageSubscriberIF[ProgressUpdate]): The progress subscriber
            results_subscriber (MessageSubscriberIF[EvaluationResultBatch]): The results subscriber
            global_rank (int): The global rank of the current process
            local_rank (int): The local rank of the current process on the current node

        Returns:
            tuple[MessagePublisher[EvaluationResultBatch], MessagePublisher[ProgressUpdate]]: The evaluation
                result publisher and the progress publisher
        """
        message_broker = MessageBroker()
        progress_publisher = MessagePublisher[ProgressUpdate](
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

        return evaluation_result_publisher, progress_publisher
