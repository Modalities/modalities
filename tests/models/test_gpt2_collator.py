import pytest
import torch

from modalities.models.gpt2.collator import GPT2LLMCollateFn


def test_gpt2_collate_shifts_samples_and_targets():
    collator = GPT2LLMCollateFn(sample_key="input_ids", target_key="labels")
    batch = [
        {"input_ids": torch.tensor([1, 2, 3, 4])},
        {"input_ids": torch.tensor([5, 6, 7, 8])},
    ]

    result = collator(batch)

    assert result.samples["input_ids"].tolist() == [[1, 2, 3], [5, 6, 7]]
    assert result.targets["labels"].tolist() == [[2, 3, 4], [6, 7, 8]]


def test_gpt2_collate_sub_seq_lengths_without_eos():
    collator = GPT2LLMCollateFn(
        sample_key="input_ids",
        target_key="labels",
        sub_seq_lengths_key="sub_seq_lengths",
        eos_token_id=99,
    )
    batch = [
        {"input_ids": torch.tensor([10, 11, 12, 13, 14])},
        {"input_ids": torch.tensor([20, 21, 22, 23, 24])},
    ]

    result = collator(batch)

    assert result.samples["sub_seq_lengths"] == [[4], [4]]


def test_gpt2_collate_sub_seq_lengths_with_eos():
    collator = GPT2LLMCollateFn(
        sample_key="input_ids",
        target_key="labels",
        sub_seq_lengths_key="sub_seq_lengths",
        eos_token_id=99,
    )
    batch = [
        {"input_ids": torch.tensor([1, 99, 2, 3, 99])},
        {"input_ids": torch.tensor([7, 8, 9, 99, 10])},
    ]

    result = collator(batch)

    assert result.samples["sub_seq_lengths"] == [[2, 2], [4]]


def test_gpt2_collate_sub_seq_lengths_with_eos_and_padding():
    collator = GPT2LLMCollateFn(
        sample_key="input_ids",
        target_key="labels",
        sub_seq_lengths_key="sub_seq_lengths",
        eos_token_id=99,
        padding_token_id=0,
    )
    batch = [
        {"input_ids": torch.tensor([1, 99, 2, 3, 4, 5])},
        {"input_ids": torch.tensor([7, 8, 99, 0, 0, 0])},
    ]

    result = collator(batch)

    assert result.samples["sub_seq_lengths"] == [[2, 3], [3]]


def test_gpt2_collate_sub_seq_lengths_adds_tail_when_not_padding():
    collator = GPT2LLMCollateFn(
        sample_key="input_ids",
        target_key="labels",
        sub_seq_lengths_key="sub_seq_lengths",
        eos_token_id=5,
        padding_token_id=0,
    )
    batch = [{"input_ids": torch.tensor([1, 5, 9, 8])}]

    result = collator(batch)

    assert result.samples["sub_seq_lengths"] == [[2, 1]]


def test_gpt2_collate_raises_when_sequence_starts_with_padding_and_no_eos():
    collator = GPT2LLMCollateFn(
        sample_key="input_ids",
        target_key="labels",
        sub_seq_lengths_key="sub_seq_lengths",
        eos_token_id=99,
        padding_token_id=0,
    )
    batch = [{"input_ids": torch.tensor([0, 1, 2, 3])}]

    with pytest.raises(AssertionError, match="Sequence starts with padding token"):
        collator(batch)
