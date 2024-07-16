from typing import Dict, List

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.config.pydanctic_if_types import PydanticCollateFnIFType
from modalities.models.gpt2.collator import CollateFnIF


class LossMaskingCollateFnWrapperConfig(BaseModel):
    collate_fn: PydanticCollateFnIFType
    target_keys_to_mask: List[str]
    loss_ignore_index: int


class LossMaskingCollateFnWrapper(CollateFnIF):
    def __init__(self, collate_fn: CollateFnIF, target_keys_to_mask: List[str], loss_ignore_index: int):
        self.collate_fn = collate_fn
        self.target_keys_to_mask = target_keys_to_mask
        self.loss_ignore_index = loss_ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        dataset_batch = self.collate_fn(batch)
        for target_key_to_mask in self.target_keys_to_mask:
            target = dataset_batch.targets[target_key_to_mask]
            # FIXME: hard coded masked token ids! This should be propageted from the data header information
            masked_target = self._mask_target(
                target, b_mask_token_id=0, e_mask_token_id=1, loss_ignore_index=self.loss_ignore_index
            )
            dataset_batch.targets[target_key_to_mask] = masked_target
        return dataset_batch

    def _mask_target(
        self, target: torch.Tensor, b_mask_token_id: int, e_mask_token_id: int, loss_ignore_index: int
    ) -> torch.Tensor:
        # FIXME replace debug target
        debug_target = [5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0, 1]
        target = torch.Tensor([debug_target, debug_target])
        assert b_mask_token_id != e_mask_token_id, "b_mask_token_id and e_mask_token_id must be different!"
        assert b_mask_token_id in target, "b_mask_token_id not found in target"
        assert e_mask_token_id in target, "e_mask_token_id not found in target"
        mask = torch.zeros_like(target)
        mask += torch.where(target != b_mask_token_id, 0, 1)
        mask += torch.where(target != e_mask_token_id, 0, -1)
        mask = mask.cumsum(-1)
        mask = mask.roll(shifts=1, dims=-1)
        mask[:, 0] = 0
        new_target = torch.where(mask > 0, target, -100)
        # TODO write test for this
        return new_target
