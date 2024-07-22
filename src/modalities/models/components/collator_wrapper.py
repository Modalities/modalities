from typing import Dict, List

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.config.pydanctic_if_types import PydanticCollateFnIFType, PydanticTokenizerIFType
from modalities.models.gpt2.collator import CollateFnIF
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


class MaskingTokenConfig(BaseModel):
    b_include_to_loss_token: str
    e_include_to_loss_token: str


class LossMaskingCollateFnWrapperConfig(BaseModel):
    collate_fn: PydanticCollateFnIFType
    target_keys_to_mask: List[str]
    loss_ignore_index: int
    special_tokens: MaskingTokenConfig
    tokenizer: PydanticTokenizerIFType


class LossMaskingCollateFnWrapper(CollateFnIF):
    def __init__(
        self,
        collate_fn: CollateFnIF,
        target_keys_to_mask: List[str],
        loss_ignore_index: int,
        special_tokens: MaskingTokenConfig,
        tokenizer: TokenizerWrapper,
    ):
        self.collate_fn = collate_fn
        self.target_keys_to_mask = target_keys_to_mask
        self.loss_ignore_index = loss_ignore_index
        self.tokenizer = tokenizer
        self.b_mask_token_id = self.tokenizer.get_token_id(special_tokens.b_include_to_loss_token)
        self.e_mask_token_id = self.tokenizer.get_token_id(special_tokens.e_include_to_loss_token)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        dataset_batch = self.collate_fn(batch)
        for target_key_to_mask in self.target_keys_to_mask:
            target = dataset_batch.targets[target_key_to_mask]
            masked_target = self._mask_target(
                target=target,
                b_mask_token_id=self.b_mask_token_id,
                e_mask_token_id=self.e_mask_token_id,
                loss_ignore_index=self.loss_ignore_index,
            )
            dataset_batch.targets[target_key_to_mask] = masked_target
        return dataset_batch

    def _mask_target(
        self, target: torch.Tensor, b_mask_token_id: int, e_mask_token_id: int, loss_ignore_index: int
    ) -> torch.Tensor:
        error_msg = ""

        if b_mask_token_id == e_mask_token_id:
            error_msg += "b_mask_token_id and e_mask_token_id must be different! "
        if b_mask_token_id not in target:
            error_msg += "b_mask_token_id not found in target "
        if e_mask_token_id not in target:
            error_msg += "e_mask_token_id not found in target "
        if error_msg:
            raise ValueError(
                "Error in masking tokens for loss computation."
                + "Make sure the tokenizer tokenized as expected. Frequent source of error: ' <token>' and '<token>'"
                + "Please check the following: "
                + error_msg
                + error_msg
            )
        mask = torch.zeros_like(target)
        mask += torch.where(target != b_mask_token_id, 0, 1)
        mask += torch.where(target != e_mask_token_id, 0, -1)

        # in case -1 is before 1 we need to include the first tokens to the loss
        end_before_begin = torch.argmax(mask, dim=-1, keepdim=True) > torch.argmin(mask, dim=-1, keepdim=True)
        mask[:, 0] = end_before_begin.squeeze()

        # mark all tokens beween 1 and -1 with 1
        mask = mask.cumsum(-1)

        # shift the mask to the right, to conform to the shifted target
        mask = mask.roll(shifts=1, dims=-1)
        mask[:, 0] = end_before_begin.squeeze()

        # apply mask: if mask is 1, keep the target, otherwise replace with loss_ignore_index
        new_target = torch.where(mask > 0, target, loss_ignore_index)
        return new_target
