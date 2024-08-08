import pytest
import torch
from modalities.batch import DatasetBatch
from modalities.models.huggingface.collator import (
    SpanMaskingCollateFn,
    SpanMaskingCollateFnConfig,
)
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer
from modalities.config.config import PreTrainedHFTokenizerConfig
def tokenize(word: str):
    vocab = {"begin": 0, "end": 1}
    return vocab[word]
@pytest.fixture
def hf_tokenizer_config() -> PreTrainedHFTokenizerConfig:
    return dict(
        pretrained_model_name_or_path="google/long-t5-tglobal-base",
        truncation=False,
        padding=False,
    )
@pytest.fixture
def dummy_tokenizer(hf_tokenizer_config):
    # get the tokenizer
    return PreTrainedHFTokenizer(**hf_tokenizer_config)
@pytest.fixture
def span_masking_config(dummy_tokenizer) -> SpanMaskingCollateFnConfig:
    return dict(
        sample_key="sample",
        target_key="target",
        noise_density=0.3,
        mean_noise_span_length=3.0,
        tokenizer=dummy_tokenizer,
    )
@pytest.mark.parametrize(
    "batch",
    [
        (
            # the collate_fn will shift the sample and target:
            {"sample": torch.arange(3, 20).int()},
        ),
    ],
)
def test_span_masking_collator(span_masking_config, batch):
    span_masking_collator = SpanMaskingCollateFn(**span_masking_config)
    result_batch = span_masking_collator(batch)
    input_ids = batch[0]["sample"]
    sample_ids = result_batch.samples["sample"]
    target_ids = result_batch.targets["target"]
    # Masked sample and masked target are shorter than input
    assert sample_ids.size()[-1] < input_ids.size()[-1]
    assert target_ids.size()[-1] < input_ids.size()[-1]
    # Masked sample and masked target end with EOS token
    assert sample_ids[:, -1].all() == span_masking_config["tokenizer"].tokenizer.eos_token_id
    assert target_ids[:, -1].all() == span_masking_config["tokenizer"].tokenizer.eos_token_id
    # Text tokens are ascending (same order as input)
    assert (torch.diff(sample_ids[(sample_ids > 1) & (sample_ids < 32000)]) >= 1).all()
    assert (torch.diff(target_ids[(target_ids > 1) & (target_ids < 32000)]) >= 1).all()
    # Special mask tokens are descending 
    assert(torch.diff(sample_ids[sample_ids >= 32000]) == -1).all()
    assert(torch.diff(target_ids[target_ids >= 32000]) == -1).all()
    # All input tokens are either in sample or target
    joined_ids = torch.concatenate(
        (
            sample_ids[(sample_ids > 1) & (sample_ids < 32000)],
            target_ids[(target_ids > 1) & (target_ids < 32000)]
        )
    )
    sorted_join_ids, _ = torch.sort(joined_ids)
    assert torch.equal(sorted_join_ids, input_ids)
