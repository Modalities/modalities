from unittest.mock import MagicMock

import pytest
import torch

from modalities.batch import DatasetBatch
from modalities.models.components.collator_wrapper import (
    LossMaskingCollateFnWrapper,
    LossMaskingCollateFnWrapperConfig,
    MaskingTokenConfig,
)
from modalities.models.gpt2.collator import GPT2LLMCollateFn
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


def tokenize(word: str):
    vocab = {"begin": 0, "end": 1}
    return vocab[word]


@pytest.fixture
def dummy_tokenizer():
    mock = MagicMock(spec=TokenizerWrapper)
    mock.get_token_id.side_effect = tokenize
    return mock


@pytest.fixture
def loss_masking_config(dummy_tokenizer) -> LossMaskingCollateFnWrapperConfig:
    return dict(
        collate_fn=GPT2LLMCollateFn(sample_key="sample", target_key="target"),
        target_keys_to_mask=["target"],
        loss_ignore_index=-100,
        mask_tokens=MaskingTokenConfig(b_include_to_loss_token="begin", e_include_to_loss_token="end"),
        tokenizer=dummy_tokenizer,
    )


# calculating nce_loss for two randomly generated batch of embeddings (manually calculated)
@pytest.mark.parametrize(
    "batch,expected_batch",
    [
        (
            [
                # the collate_fn will shift the sample and target:
                # shifted sample:           [5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0]
                # shifted target:           [5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0, 1]
                # masked shifted target:    [-100, -100, 5, 5, -100, -100, -100, 5, -100, -100, -100, -100, -100, -100]
                {"sample": torch.Tensor([5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0, 1])},
                # shifted sample:           [5, 5, 1, 5, 0, 5, 5, 5, 1, 5, 5, 5, 5, 5]
                # shifted target:           [5, 1, 5, 0, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5]
                # masked shifted target:    [5, -100, -100, -100, 5, 5, 5, -100, -100, -100, -100, -100, -100, -100]
                {"sample": torch.Tensor([5, 5, 1, 5, 0, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5])},
            ],
            # the expected batch is shifted and masked for loss computation!
            DatasetBatch(
                targets={
                    "target": torch.Tensor(
                        [
                            # expected case
                            [-100, -100, 5, 5, -100, -100, -100, 5, -100, -100, -100, -100, -100, -100],
                            # case: if dataset splits the assisstant role across batches, Keep those tokens at the front
                            [5, -100, -100, -100, 5, 5, 5, -100, -100, -100, -100, -100, -100, -100],
                        ]
                    )
                },
                samples={
                    # not needed for the test
                },
            ),
        )
    ],
)
def test_loss_masking(loss_masking_config, batch, expected_batch):
    loss_masking_collator = LossMaskingCollateFnWrapper(**loss_masking_config)
    result_batch = loss_masking_collator(batch)
    assert torch.equal(result_batch.targets["target"], expected_batch.targets["target"])
