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
    mask_tokens: MaskingTokenConfig
    tokenizer: PydanticTokenizerIFType


class LossMaskingCollateFnWrapper(CollateFnIF):
    def __init__(
        self,
        collate_fn: CollateFnIF,
        target_keys_to_mask: List[str],
        loss_ignore_index: int,
        mask_tokens: MaskingTokenConfig,
        tokenizer: TokenizerWrapper,
    ):
        """Wraps the given collate_fn and masks the target keys if not within the given special mask tokens.
        Does not include both mask tokens into the loss. If you need a token to indicate the end of the assistant,
        use another special token for this!
        Works also for the continuous dataset reading, as if the "end-include-to-loss" token is detected in the front,
        all tokens before are included to the loss.

        Throws a ValueError if the mask tokens are not found in the target or if the mask tokens are the same.
        """
        self.collate_fn = collate_fn
        self.target_keys_to_mask = target_keys_to_mask
        self.loss_ignore_index = loss_ignore_index
        self.tokenizer = tokenizer
        self.b_mask_token_id = self.tokenizer.get_token_id(mask_tokens.b_include_to_loss_token)
        self.e_mask_token_id = self.tokenizer.get_token_id(mask_tokens.e_include_to_loss_token)

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
        # we shift the mask to the right, to exclude not only the end mask token but also
        # the begin mask token from the loss
        mask[:, 1:] += torch.where(target != b_mask_token_id, 0, 1)[:, :-1]
        mask += torch.where(target != e_mask_token_id, 0, -1)

        # in case -1 (end mask token indicator) is before 1 (begin mask token indicator) we need to
        # include the first tokens to the loss
        end_before_begin = torch.argmax(mask, dim=-1, keepdim=True) > torch.argmin(mask, dim=-1, keepdim=True)
        mask[:, 0] = end_before_begin.squeeze()

        # mark all tokens beween 1 (begin mask token indicator) and -1 (end mask token indicator) with 1
        # this includes the 1, but due to the shift above, we exclude both!
        include_to_loss_mask = mask.cumsum(-1)

        # apply mask: if mask is 1, keep the target, otherwise replace with loss_ignore_index
        new_target = torch.where(include_to_loss_mask.bool(), target, loss_ignore_index)
        return new_target
