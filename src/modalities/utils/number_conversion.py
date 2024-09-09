import re
from pathlib import Path
from typing import Annotated, Callable

from pydantic import BaseModel, Field


class LocalNumBatchesFromNumSamplesConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    global_num_samples: Annotated[int, Field(strict=True, ge=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]


class LocalNumBatchesFromNumTokensConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    global_num_tokens: Annotated[int, Field(strict=True, ge=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]


class NumStepsFromNumSamplesConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    global_num_samples: Annotated[int, Field(strict=True, ge=0)]


class NumStepsFromNumTokensConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    global_num_tokens: Annotated[int, Field(strict=True, ge=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]


class NumTokensFromNumStepsConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]


class LastStepFromCheckpointPathConfig(BaseModel):
    checkpoint_path: Path


class GlobalNumSeenTokensFromCheckpointPathConfig(BaseModel):
    checkpoint_path: Path


class NumStepsFromNumTokensAndCheckpointPathConfig(BaseModel):
    checkpoint_path: Path
    global_num_tokens: Annotated[int, Field(strict=True, ge=0)]


class NumberConversion:
    @staticmethod
    def get_local_num_batches_from_num_samples(
        num_ranks: int, global_num_samples: int, local_micro_batch_size: int
    ) -> int:
        """Calculates the number of local batches for each rank, given the global
        number of samples and number of ranks.
        This helper function is primarily used to calculate the number of batches to
        skip when resuming a dataloader during warmstart.

        Args:
            num_ranks (int): Global number of ranks.
            global_num_samples (int): Global number of samples.
            local_micro_batch_size (int): Local micro batch size on single rank.

        Returns:
            int: Number of local batches for single rank.
        """
        return (global_num_samples) // (num_ranks * local_micro_batch_size)

    @staticmethod
    def get_local_num_batches_from_num_tokens(
        num_ranks: int, global_num_tokens: int, sequence_length: int, local_micro_batch_size: int
    ) -> int:
        """Calculates the number of local batches for each rank, given the global
        number of tokens and number of ranks.
        This helper function is primarily used to calculate a dataloader's number of batches (total and to skip)

        Args:
            num_ranks (int): Global number of ranks.
            global_num_tokens (int): Global number of tokens.
            sequence_length (int): Sequence length of the model.
            local_micro_batch_size (int): Local micro batch size on single rank.
        Returns:
            int: Number of local batches for single rank.
        """
        global_num_samples = global_num_tokens // sequence_length
        return NumberConversion.get_local_num_batches_from_num_samples(
            num_ranks=num_ranks, global_num_samples=global_num_samples, local_micro_batch_size=local_micro_batch_size
        )

    @staticmethod
    def get_num_steps_from_num_samples(num_ranks: int, local_micro_batch_size: int, global_num_samples: int) -> int:
        """Calculates the number of steps given the global
        number of samples, local micro batch size and number of ranks.

        Args:
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            global_num_samples (int): Global number of samples.

        Returns:
            int: Number of steps.
        """
        return global_num_samples // (num_ranks * local_micro_batch_size)

    @staticmethod
    def get_num_steps_from_num_tokens(
        num_ranks: int, local_micro_batch_size: int, global_num_tokens: int, sequence_length: int
    ) -> int:
        """Calculates the number of steps given the global
        number of tokens, local micro batch size and number of ranks.
        This helper function is primarily used to calculate the number of batches to
        skip when resuming a dataloader during warmstart.

        Args:
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            global_num_tokens (int): Global number of tokens.
            sequence_length (int): Sequence length of the model.

        Returns:
            int: Number of steps.
        """
        global_num_samples = global_num_tokens // sequence_length
        return NumberConversion.get_num_steps_from_num_samples(
            num_ranks=num_ranks, local_micro_batch_size=local_micro_batch_size, global_num_samples=global_num_samples
        )

    @staticmethod
    def get_num_tokens_from_num_steps_callable(
        num_ranks: int, local_micro_batch_size: int, sequence_length: int
    ) -> Callable[[int], int]:
        """Returns a callable that calculates the number of global tokens given the number of steps done.

        Args:
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            sequence_length (int): Sequence length of the model.

        Returns:
            Callable[[int], int]: Callable that calculates the number of global tokens.
        """
        return lambda num_steps_done: num_steps_done * num_ranks * local_micro_batch_size * sequence_length

    @staticmethod
    def get_last_step_from_checkpoint_path(checkpoint_path: Path) -> int:
        """Returns the last step from the checkpoint path.

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            int: Last step from the checkpoint path.
        """
        # Regex pattern to match 'num_steps_' followed by digits
        pattern = r"num_steps_(\d+)"
        match = re.search(pattern, str(checkpoint_path))

        # Extract the number of steps if a match is found
        if match:
            num_steps = int(match.group(1))  # Group 1 contains the digits after 'num_steps_'
        else:
            raise ValueError(f"No match found for pattern {pattern} in {checkpoint_path}")
        return num_steps - 1

    @staticmethod
    def get_global_num_seen_tokens_from_checkpoint_path(checkpoint_path: Path) -> int:
        """Returns the global num seen tokens from the checkpoint path.

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            int: Num seen tokens from the checkpoint path.
        """
        # Regex pattern to match 'num_steps_' followed by digits
        pattern = r"num_tokens_(\d+)"
        match = re.search(pattern, str(checkpoint_path))

        # Extract the number of steps if a match is found
        if match:
            num_tokens = int(match.group(1))  # Group 1 contains the digits after 'num_tokens_'
        else:
            raise ValueError(f"No match found for pattern {pattern} in {checkpoint_path}")
        return num_tokens

    @staticmethod
    def get_num_steps_from_num_tokens_and_checkpoint_path(checkpoint_path: Path, global_num_tokens: int) -> int:
        tokens_per_step = NumberConversion.get_global_num_seen_tokens_from_checkpoint_path(checkpoint_path) / (
            NumberConversion.get_last_step_from_checkpoint_path(checkpoint_path) + 1
        )
        num_steps = global_num_tokens // tokens_per_step
        if isinstance(num_steps, float) and not num_steps.is_integer():
            raise ValueError(f"Number of steps calculated is not an integer. {num_steps}")
        return int(num_steps)
