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
        special_tokens=MaskingTokenConfig(b_include_to_loss_token="begin", e_include_to_loss_token="end"),
        tokenizer=dummy_tokenizer,
    )


# calculating nce_loss for two randomly generated batch of embeddings (manually calculated)
@pytest.mark.parametrize(
    "batch,expected_batch",
    [
        (
            [
                {"sample": torch.Tensor([5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0, 1])},
                {"sample": torch.Tensor([5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0, 1])},
            ],
            # the expected batch is shifted and masked for loss computation!
            DatasetBatch(
                targets={
                    "target": torch.Tensor(
                        [
                            [-100, -100, 5, 5, 1, -100, -100, 5, 1, -100, 1, -100, -100, 1],
                            [-100, -100, 5, 5, 1, -100, -100, 5, 1, -100, 1, -100, -100, 1],
                        ]
                    )
                },
                samples={
                    "sample": torch.Tensor(
                        [[5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0], [5, 5, 0, 5, 5, 1, 5, 0, 5, 1, 0, 1, 5, 0]]
                    )
                },
            ),
        )
    ],
)
def test_loss_masking(loss_masking_config, batch, expected_batch):
    loss_masking_collator = LossMaskingCollateFnWrapper(**loss_masking_config)
    result_batch = loss_masking_collator(batch)
    assert torch.equal(result_batch.targets["target"], expected_batch.targets["target"])
