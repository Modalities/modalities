from typing import List

import torch
from pydantic import BaseModel

from modalities.config.pydantic_if_types import PydanticTokenizerIFType
from modalities.dataloader.collate_fns.collate_if import CollateFnIF
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.util import warn_rank_0


class LossMaskingTokenConfig(BaseModel):
    b_include_to_loss_token: str
    e_include_to_loss_token: str


class LossMaskingCollateFnConfig(BaseModel):
    target_keys_to_mask: List[str]
    loss_ignore_index: int
    mask_tokens: LossMaskingTokenConfig
    tokenizer: PydanticTokenizerIFType


class LossMaskingCollateFn(CollateFnIF):
    def __init__(
        self,
        target_keys_to_mask: List[str],
        loss_ignore_index: int,
        mask_tokens: LossMaskingTokenConfig,
        tokenizer: TokenizerWrapper,
    ):
        """
        Initializes the LossMaskingCollateFnWrapper.
        The colate function masks the target keys if not within the given special mask tokens.
        Does not include both mask tokens into the loss. If you need a token to indicate the end of the assistant,
        use another special token for this!
        Works also for the continuous dataset reading, as if the "end-include-to-loss" token is detected in the front,
        all tokens before are included to the loss.

        Throws a ValueError if the mask tokens are not found in the target or if the mask tokens are the same.


        Args:
            target_keys_to_mask (List[str]): The list of target keys to mask.
            loss_ignore_index (int): The index to ignore in the loss calculation.
            mask_tokens (MaskingTokenConfig): Entails begin and end tokens, which mark (exclusive) inclusion to the
            loss.
            tokenizer (TokenizerWrapper): The tokenizer wrapper.

        Raises:
            ValueError: If b_mask_token_id and e_mask_token_id are the same.
        """
        self.target_keys_to_mask = target_keys_to_mask
        self.loss_ignore_index = loss_ignore_index
        self.tokenizer = tokenizer
        self.b_mask_token_id = self.tokenizer.get_token_id(mask_tokens.b_include_to_loss_token)
        self.e_mask_token_id = self.tokenizer.get_token_id(mask_tokens.e_include_to_loss_token)
        if self.b_mask_token_id == self.e_mask_token_id:
            raise ValueError(
                "b_mask_token_id and e_mask_token_id of the LossMaskingCollateFnWrapper must be different!"
            )

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Collates a batch of data by applying target masking.

        Args:
            batch (dict[str, torch.Tensor]): The batch contains keys corresponding to different
                data modalities and their
            respective tensors.

        Returns:
            dict[str, torch.Tensor]: A batch dict with masked targets.

        """

        for target_key_to_mask in self.target_keys_to_mask:
            target = batch[target_key_to_mask]
            masked_target = self._mask_target(
                target=target,
                b_mask_token_id=self.b_mask_token_id,
                e_mask_token_id=self.e_mask_token_id,
                loss_ignore_index=self.loss_ignore_index,
            )
            batch[target_key_to_mask] = masked_target
        return batch

    def _mask_target(
        self, target: torch.Tensor, b_mask_token_id: int, e_mask_token_id: int, loss_ignore_index: int
    ) -> torch.Tensor:
        """
        We mask the target tensor with loss_ignore_index between, but not inclusive the begin and end mask token.
        We do this vectorizes, as this is fast.
        Example:
            sample_orig =      [2,2,3,2, 2,4,2,2,2]
            sample =           [2,2,3,2, 2,4,2,2] # from collate_fn
            target =           [2,3,2,2, 4,2,2,2] # from collate_fn
            mask_initially =   [0,0,0,0, 0,0,0,0] # mask = torch.zeros_like(target)
            mask_shifted_1 =   [0,0,1,0, 0,0,0,0] # mask[:, 1:] += torch.where(target != b_mask_token_id, 0, 1)[:, :-1]
            mask_shifted_2 =   [0,0,1,0,-1,0,0,0] # mask += torch.where(target != e_mask_token_id, 0, -1)
            mask_cumsum =      [0,0,1,1, 0,0,0,0] # include_to_loss_mask = mask.cumsum(-1)


        By shifting only the b_mask_token_id to the right, we exclude the begin mask token from the loss, as otherwise
        cumsum would include the begin mask token. Example without shift:
            mask_no_shift_2    [0,1,0,0,-1,0,0,0]
            cumsum_no_shift    [0,1,1,1, 0,0,0,0]

        If the b_mask_token_id is not found in the target tensor, we skip the sample.

        Args:
            target (torch.Tensor): The target tensor to be masked.
            b_mask_token_id (int): The token ID indicating the beginning of the mask.
            e_mask_token_id (int): The token ID indicating the end of the mask.
            loss_ignore_index (int): The index to replace masked tokens with.

        Returns:
            torch.Tensor: The masked target tensor.

        Raises:
            ValueError: If the end mask token indicator is before the begin mask token indicator in the target tensor.
            ValueError: If the masking tokens are not alternating in the target tensor.
        """
        if b_mask_token_id not in target:
            warn_rank_0(
                "During masking tokens for loss computation, b_mask_token_id not found in target. "
                + "Make sure the tokenizer tokenizes as expected. "
                + "Frequent source of error is the tokenization of spaces: "
                + "e.g. ' <token>' and '<token>' are different tokens. "
                + "Another reason for this error could be the first user query might take up all context, "
                + "before the assistant turn appears. Increase the context size or check your data. "
                + "We skip this sample."
            )
            return torch.ones_like(target) * loss_ignore_index
        if e_mask_token_id not in target:
            warn_rank_0(
                "During masking tokens for loss computation, e_mask_token_id not found in target. "
                + "Make sure the tokenizer tokenizes as expected. "
                + "Frequent source of error is the tokenization of spaces: "
                + "e.g. ' <token>' and '<token>' are different tokens. "
                + "We skip this sample."
            )
            return torch.ones_like(target) * loss_ignore_index

        mask = torch.zeros_like(target)
        # we shift the mask to the right, to exclude not only the end mask token but also
        # the begin mask token from the loss
        mask[:, 1:] += torch.where(target != b_mask_token_id, 0, 1)[:, :-1]
        mask += torch.where(target != e_mask_token_id, 0, -1)

        # mark all tokens beween 1 (begin mask token indicator) and -1 (end mask token indicator) with 1
        # this includes the -1, but due to the shift above, we exclude both!
        include_to_loss_mask = mask.cumsum(-1)

        # check that the sequence has alternating start and end mask token indicators starting with a start mask token
        # we explicitly allow ending on a start mask token
        if not ((0 <= include_to_loss_mask).all() and (include_to_loss_mask <= 1).all()):
            raise ValueError(
                "end mask token indicator is before begin mask token indicator in the target. "
                + "This is not supported by the LossMaskingCollateFn."
            )

        # apply mask: if mask is 1, keep the target, otherwise replace with loss_ignore_index
        new_target = torch.where(include_to_loss_mask.bool(), target, loss_ignore_index)
        return new_target
