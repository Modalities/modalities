from typing import Annotated, Callable

from pydantic import BaseModel, Field


class LocalNumBatchesFromNumSamplesConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    global_num_samples: Annotated[int, Field(strict=True, ge=0)]


class LocalNumBatchesFromNumTokensConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    global_num_tokens: Annotated[int, Field(strict=True, ge=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]


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


class NumberConversion:
    @staticmethod
    def get_local_num_batches_from_num_samples(num_ranks: int, global_num_samples: int) -> int:
        """Calculates the number of local batches for each rank, given the global
        number of samples and number of ranks.
        This helper function is primarily used to calculate the number of batches to
        skip when resuming a dataloader during warmstart.

        Args:
            num_ranks (int): _description_
            global_num_samples (int): _description_

        Returns:
            int: _description_
        """
        return global_num_samples // num_ranks

    @staticmethod
    def get_local_num_batches_from_num_tokens(num_ranks: int, global_num_tokens: int, sequence_length: int) -> int:
        """Calculates the number of local batches for each rank, given the global
        number of tokens and number of ranks.
        This helper function is primarily used to calculate a dataloader's number of batches (total and to skip)

        Args:
            num_ranks (int): _description_
            global_num_tokens (int): _description_
            sequence_length (int): _description_

        Returns:
            int: _description_
        """
        global_num_samples = global_num_tokens // sequence_length
        return NumberConversion.get_local_num_batches_from_num_samples(
            num_ranks=num_ranks, global_num_samples=global_num_samples
        )

    @staticmethod
    def get_num_steps_from_num_samples(num_ranks: int, local_micro_batch_size: int, global_num_samples: int) -> int:
        """Calculates the number of steps given the global
        number of samples, local micro batch size and number of ranks.

        Args:
            num_ranks (int): _description_
            local_micro_batch_size (int): _description_
            global_num_samples (int): _description_

        Returns:
            int: _description_
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
            num_ranks (int): _description_
            local_micro_batch_size (int): _description_
            global_num_tokens (int): _description_
            sequence_length (int): _description_

        Returns:
            int: _description_
        """
        global_num_samples = global_num_tokens // sequence_length
        return NumberConversion.get_num_steps_from_num_samples(
            num_ranks=num_ranks, local_micro_batch_size=local_micro_batch_size, global_num_samples=global_num_samples
        )

    @staticmethod
    def get_num_tokens_from_num_steps_callable(
        num_ranks: int, local_micro_batch_size: int, sequence_length: int
    ) -> Callable[[int], int]:
        return lambda num_steps_done: num_steps_done * num_ranks * local_micro_batch_size * sequence_length
