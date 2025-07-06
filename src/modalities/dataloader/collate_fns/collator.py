from typing import Optional

import torch
from pydantic import BaseModel, model_validator

from modalities.batch import DatasetBatch
from modalities.config.pydantic_if_types import PydanticCollateFnIFType
from modalities.dataloader.collate_fns.collate_if import CollateFnIF, CollatorIF


class DefaultWrappingCollatorConfig(BaseModel):
    input_keys: list[str]
    sample_keys: list[str]
    target_keys: list[str]
    collate_fns: list[PydanticCollateFnIFType] = None
    sequence_length: int = None
    padding_token_id: int = None

    @model_validator(mode="after")
    def validate_sequence_length_and_padding(self):
        if self.sequence_length is None != self.padding_token_id is None:
            raise ValueError("If sequence_length is set, padding_token_id must also be set.")
        return self


class DefaultWrappingCollator(CollatorIF):
    """DefaultWrappingCollator class to define a collate function that pads and
    truncates sequences to a fixed length and applies passed collate functions
    sequentially."""

    def __init__(
        self,
        input_keys: list[str],
        sample_keys: list[str],
        target_keys: list[str],
        collate_fns: Optional[list[CollateFnIF]] = None,
        sequence_length: Optional[int] = None,
        padding_token_id: Optional[int] = None,
    ):
        """
        Initializes the Collator object.

        Args:
            input_keys (list[str]): List of keys for the input data.
            sample_keys (list[str]): List of keys for the resulting sample data.
            target_keys (list[str]): List of keys for the resulting target data.
            collate_fns (list[CollateFnIF], optional): List of wrapped collate functions to apply sequentially.
                Defaults to None.
            sequence_length (int, optional): Fixed sequence length for padding/truncating. Defaults to None.
            padding_token_id (int, optional): Token ID used for padding. Defaults to None.

        Raises:
            ValueError: If sequence_length is set but padding_token_id is not set or vice versa.
        """
        self.input_keys = input_keys
        self.sampple_keys = sample_keys
        self.target_keys = target_keys
        self.collate_fns = collate_fns if collate_fns is not None else []
        self.sequence_length = sequence_length
        self.padding_token_id = padding_token_id
        if sequence_length is None != padding_token_id is None:
            raise ValueError("If sequence_length is set, padding_token_id must also be set.")

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
        """Process a batch of data by calling the wrapped collate function.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing 1-dim tensors.

        Returns:
            DatasetBatch: The processed batch of data.
        """
        # convert to tensors
        batch = [{k: torch.tensor(v) for k, v in tensor_dict.items()} for tensor_dict in batch]

        if self.sequence_length is not None and self.padding_token_id is not None:
            # Pad and truncate the sequences in the batch to the fixed sequence length
            self._pad_and_truncate_inplace(batch)

        sample_tensor_dict = {key: torch.stack([d[key] for d in batch]) for key in self.sampple_keys}
        for wrapped_collate_fn in self.collate_fns:
            sample_tensor_dict = wrapped_collate_fn(sample_tensor_dict)

        samples = {sample_key: sample_tensor_dict[sample_key] for sample_key in self.sampple_keys}
        targets = {target_key: sample_tensor_dict[target_key] for target_key in self.target_keys}
        return DatasetBatch(targets=targets, samples=samples)

    def _pad_and_truncate_inplace(self, batch: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        for sample in batch:
            for key in sample.keys():
                seq = sample[key]
                if seq.dim() != 1:
                    raise ValueError(
                        f"Expected tensor with at least one dimension, got {seq.dim()} dimensions for key '{key}'."
                    )

                # Truncate or pad to fixed sequence length
                if seq.size(0) > self.sequence_length:
                    seq = seq[: self.sequence_length]
                elif seq.size(0) < self.sequence_length:
                    padding = torch.full((self.sequence_length - seq.size(0),), self.padding_token_id, dtype=seq.dtype)
                    seq = torch.cat([seq, padding], dim=0)
                sample[key] = seq
