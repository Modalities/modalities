import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from modalities.dataloader.dataset_factory import DatasetFactory


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
    gradient_accumulation_steps: Annotated[int, Field(strict=True, gt=0)]


class NumStepsFromNumTokensConfig(BaseModel):
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    global_num_tokens: Annotated[int, Field(strict=True, ge=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(strict=True, gt=0)]


class NumTokensFromNumStepsConfig(BaseModel):
    num_steps: Annotated[int, Field(strict=True, ge=0)]
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(strict=True, gt=0)]


class NumberConversionFromCheckpointPathConfig(BaseModel):
    checkpoint_path: Path


class NumTokensFromPackedMemMapDatasetContinuousConfig(BaseModel):
    dataset_path: Path
    sequence_length: Annotated[int, Field(strict=True, gt=0)]
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(strict=True, gt=0)]


class NumStepsFromRawDatasetIndexConfig(BaseModel):
    raw_index_path: Path
    num_ranks: Annotated[int, Field(strict=True, gt=0)]
    local_micro_batch_size: Annotated[int, Field(strict=True, gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(strict=True, gt=0)]


class NumberConversion:
    @staticmethod
    def _get_checkpoint_parameter_value(pattern: str, string: str) -> int:
        matches = re.findall(pattern, string)

        # Extract the number of steps if a match is found
        if len(matches) == 1:
            value = int(matches[0])
            return value
        elif len(matches) > 1:
            raise ValueError(
                f"Expected a single group in the match. Got {len(matches)} matches: {matches}. "
                f"Pattern: {pattern}, String: {string}"
            )
        else:
            raise ValueError(f"No match found for pattern {pattern} in {string}")

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
        return (global_num_samples) // num_ranks // local_micro_batch_size

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
    def get_num_steps_from_num_samples(
        num_ranks: int, local_micro_batch_size: int, global_num_samples: int, gradient_accumulation_steps: int
    ) -> int:
        """Calculates the number of steps given the global
        number of samples, local micro batch size, number of ranks and gradient accumulation steps.

        Args:
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            global_num_samples (int): Global number of samples.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.

        Returns:
            int: Number of steps.
        """
        return global_num_samples // num_ranks // local_micro_batch_size // gradient_accumulation_steps

    @staticmethod
    def get_num_steps_from_num_tokens(
        num_ranks: int,
        local_micro_batch_size: int,
        global_num_tokens: int,
        sequence_length: int,
        gradient_accumulation_steps: int,
    ) -> int:
        """Calculates the number of steps given the global
        number of tokens, local micro batch size number of ranks and gradient accumulation steps.

        Args:
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            global_num_tokens (int): Global number of tokens.
            sequence_length (int): Sequence length of the model.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.

        Returns:
            int: Number of steps.
        """
        global_num_samples = global_num_tokens // sequence_length
        return NumberConversion.get_num_steps_from_num_samples(
            num_ranks=num_ranks,
            local_micro_batch_size=local_micro_batch_size,
            global_num_samples=global_num_samples,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    @staticmethod
    def get_num_tokens_from_num_steps(
        num_steps: int,
        num_ranks: int,
        local_micro_batch_size: int,
        sequence_length: int,
        gradient_accumulation_steps: int,
    ) -> int:
        """Calculates the number of global tokens given the number of steps, number of ranks, local micro batch size,
            sequence length and gradient accumulation steps.

        Args:
            num_steps (int): Number of steps.
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            sequence_length (int): Sequence length of the model.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.

        Returns:
            int: Number of global tokens.
        """
        num_tokens = num_steps * num_ranks * local_micro_batch_size * sequence_length * gradient_accumulation_steps
        return num_tokens

    @staticmethod
    def get_last_step_from_checkpoint_path(checkpoint_path: Path) -> int:
        """Returns the last step from the checkpoint path.

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            int: Last step from the checkpoint path.
        """
        # Regex pattern to match 'num_steps_' followed by digits
        pattern = r"seen_steps_(\d+)"
        num_seen_steps = NumberConversion._get_checkpoint_parameter_value(pattern, str(checkpoint_path))
        return num_seen_steps - 1

    @staticmethod
    def get_num_seen_steps_from_checkpoint_path(checkpoint_path: Path) -> int:
        """Returns the number of seen steps from the checkpoint path."

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            int: Number of seen steps from the checkpoint path.
        """
        # Regex pattern to match 'num_steps_' followed by digits
        pattern = r"seen_steps_(\d+)"
        num_seen_steps = NumberConversion._get_checkpoint_parameter_value(pattern, str(checkpoint_path))
        return num_seen_steps

    @staticmethod
    def get_global_num_seen_tokens_from_checkpoint_path(checkpoint_path: Path) -> int:
        """Returns the global num seen tokens from the checkpoint path.

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            int: Num seen tokens from the checkpoint path.
        """
        # Regex pattern to match 'num_steps_' followed by digits
        pattern = r"seen_tokens_(\d+)"
        num_seen_tokens = NumberConversion._get_checkpoint_parameter_value(pattern, str(checkpoint_path))
        return num_seen_tokens

    @staticmethod
    def get_global_num_target_tokens_from_checkpoint_path(checkpoint_path: Path) -> int:
        """Returns the global num target tokens from the checkpoint path.

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            int: Num target tokens from the checkpoint path.
        """
        # Regex pattern to match 'num_steps_' followed by digits
        pattern = r"target_tokens_(\d+)"
        num_target_tokens = NumberConversion._get_checkpoint_parameter_value(pattern, str(checkpoint_path))
        return num_target_tokens

    @staticmethod
    def get_num_target_steps_from_checkpoint_path(checkpoint_path: Path) -> int:
        tokens_per_step = NumberConversion.get_global_num_seen_tokens_from_checkpoint_path(checkpoint_path) / (
            NumberConversion.get_last_step_from_checkpoint_path(checkpoint_path) + 1
        )

        global_num_target_tokens = NumberConversion.get_global_num_target_tokens_from_checkpoint_path(checkpoint_path)

        num_target_steps = global_num_target_tokens // tokens_per_step
        if isinstance(num_target_steps, float) and not num_target_steps.is_integer():
            raise ValueError(f"Number of steps calculated is not an integer. {num_target_steps}")
        return int(num_target_steps)

    @staticmethod
    def get_num_tokens_from_packed_mem_map_dataset_continuous(
        dataset_path: Path,
        sequence_length: int,
        num_ranks: int,
        local_micro_batch_size: int,
        gradient_accumulation_steps: int,
    ) -> int:
        """Get the number of tokens in a tokenized dataset that will be effectively used during training.
            Due to the way the data is packed, batched and distributed, the number of tokens used during training
            might not the same as the number of tokens in the dataset.

            The number of tokens that are used during training is calculated as follows:
                num_steps = num_dataset_tokens // sequence_length// num_ranks //
                            local_micro_batch_size // gradient_accumulation_steps
                global_num_tokens = num_steps * sequence_length * num_ranks *
                                    local_micro_batch_size * gradient_accumulation_steps


        Args:
            dataset_path (Path): Path to the tokenized dataset.
            sequence_length (int): Sequence length of the model.
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.

        Returns:
            int: Number of tokens that will be effectively used during training.
        """
        dataset = DatasetFactory.get_packed_mem_map_dataset_continuous(
            raw_data_path=dataset_path, sequence_length=sequence_length, sample_key="text"
        )
        global_num_tokens_dataset = len(dataset) * sequence_length
        num_steps = NumberConversion.get_num_steps_from_num_tokens(
            num_ranks=num_ranks,
            local_micro_batch_size=local_micro_batch_size,
            global_num_tokens=global_num_tokens_dataset,
            sequence_length=sequence_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        global_num_tokens_actual = NumberConversion.get_num_tokens_from_num_steps(
            num_ranks=num_ranks,
            local_micro_batch_size=local_micro_batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_steps=num_steps,
        )
        return global_num_tokens_actual

    @staticmethod
    def get_num_steps_from_raw_dataset_index(
        raw_index_path: Path,
        num_ranks: int,
        local_micro_batch_size: int,
        gradient_accumulation_steps: int,
    ) -> int:
        """Get the number of steps from the raw index, number of ranks, local micro batch size
        and gradient accumulation steps. The index is a list of tuples where each tuple contains
        the offset and length of a sample in the raw data.
        Note, the index is not packed and the number of samples in respective raw JSONL
        file is the same as the length of the index.

        Args:
            raw_index_path (Path): Path to the raw index file of the JSONL dataset.
            num_ranks (int): Global number of ranks.
            local_micro_batch_size (int): Local micro batch size on single rank.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.

        Returns:
            int: Number of steps.
        """
        index = DatasetFactory.get_raw_index(raw_index_path=raw_index_path)
        num_samples = len(index)
        num_steps = NumberConversion.get_num_steps_from_num_samples(
            num_ranks=num_ranks,
            local_micro_batch_size=local_micro_batch_size,
            global_num_samples=num_samples,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        return num_steps
