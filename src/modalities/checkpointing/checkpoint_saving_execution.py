from abc import ABC, abstractmethod

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction
from modalities.checkpointing.stateful.app_state import AppState
from modalities.training.training_progress import TrainingProgress


class CheckpointSavingExecutionABC(ABC):
    """Abstract class for saving PyTorch model and optimizer checkpoints."""

    @abstractmethod
    def _save_checkpoint(self, app_state: AppState, training_progress: TrainingProgress):
        """
        Saves the checkpoint of the model and optimizer.

        Args:
            model (nn.Module): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            training_progress (TrainingProgress): The training progress.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, training_progress: TrainingProgress):
        """
        Deletes the checkpoint based on the training progress.

        Args:
            training_progress (TrainingProgress): The training progress.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        training_progress: TrainingProgress,
        app_state: AppState,
    ):
        """
        Runs the checkpoint instruction.

        Args:
            checkpointing_instruction (CheckpointingInstruction): The checkpointing instruction.
            training_progress (TrainingProgress): The training progress.
            app_state (AppState): The application state to be checkpointed.
        """
        if checkpointing_instruction.save_current:
            self._save_checkpoint(app_state=app_state, training_progress=training_progress)

        for training_progress_to_delete in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(training_progress=training_progress_to_delete)
