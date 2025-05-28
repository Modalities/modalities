from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.checkpointing.checkpoint_saving_strategies import CheckpointSavingStrategyIF
from modalities.checkpointing.stateful.app_state import AppState
from modalities.training.training_progress import TrainingProgress


class CheckpointSaving:
    """Class for saving checkpoints based on a savig and execution strategy."""

    def __init__(
        self,
        checkpoint_saving_strategy: CheckpointSavingStrategyIF,
        checkpoint_saving_execution: CheckpointSavingExecutionABC,
    ):
        """
        Initializes the CheckpointSaving object.

        Args:
            checkpoint_saving_strategy (CheckpointSavingStrategyIF): The strategy for saving checkpoints.
            checkpoint_saving_execution (CheckpointSavingExecutionABC): The execution for saving checkpoints.
        """
        self.checkpoint_saving_strategy = checkpoint_saving_strategy
        self.checkpoint_saving_execution = checkpoint_saving_execution

    def save_checkpoint(
        self,
        training_progress: TrainingProgress,
        evaluation_result: dict[str, EvaluationResultBatch],
        app_state: AppState,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        """
        Saves a checkpoint of the model and optimizer.

        Args:
            training_progress (TrainingProgress): The training progress.
            evaluation_result (dict[str, EvaluationResultBatch]): The evaluation result.
            app_state (AppState): The application state to be checkpointed.
            early_stoppping_criterion_fulfilled (bool, optional):
            Whether the early stopping criterion is fulfilled. Defaults to False.
        """
        checkpointing_instruction = self.checkpoint_saving_strategy.get_checkpoint_instruction(
            training_progress=training_progress,
            evaluation_result=evaluation_result,
            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled,
        )

        self.checkpoint_saving_execution.run_checkpoint_instruction(
            checkpointing_instruction=checkpointing_instruction,
            training_progress=training_progress,
            app_state=app_state,
        )
