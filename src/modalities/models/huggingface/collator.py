from typing import Dict, List

import numpy as np
import torch
import torch.distributed
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.config.pydanctic_if_types import PydanticTokenizerIFType
from modalities.models.gpt2.collator import CollateFnIF
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


class SpanMaskingCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str
    noise_density: float
    mean_noise_span_length: float
    tokenizer: PydanticTokenizerIFType


class SpanMaskingCollateFn(CollateFnIF):
    """
    Data collator for T5 random span masked language modeling.
    Paper: <https://arxiv.org/pdf/1910.10683.pdf>
    Code taken with minor modifications from <https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py#L308>
    and ported from jax to torch.
    """

    def __init__(
        self,
        sample_key: str,
        target_key: str,
        noise_density: float,
        mean_noise_span_length: float,
        tokenizer: TokenizerWrapper,
    ):
        self.sample_key = sample_key
        self.target_key = target_key
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        """The collator prepares data for an encoder-decoder model.
        `samples` are used as encoder input, `targets` are shifted right by 1
        and used as decoder input for autoregressive modeling with teacher forcing.
        Length after masking is deterministic given noise_density and mean_noise_span_length:
        all samples will have the same length, as will all targets.


        Args:
            batch (List[Dict[str, torch.Tensor]]): batch of input token ids

        Returns:
            DatasetBatch: Contains samples and targets with random spans masked out.
        """
        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])

        batch_size, expanded_input_length = sample_tensor.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(expanded_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        input_ids = self.filter_input_ids(sample_tensor, input_ids_sentinel)
        label_ids = self.filter_input_ids(sample_tensor, labels_sentinel)

        samples = {self.sample_key: torch.from_numpy(input_ids)}
        targets = {self.target_key: torch.from_numpy(label_ids)}

        return DatasetBatch(targets=targets, samples=samples)

    def create_sentinel_ids(self, mask_indices: np.ndarray[bool]) -> np.ndarray[int]:
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        Sentinel ids are defined by the tokenizer as 32099: <extra_id_0>, 32098: <extra_id_1>, ..., 32000: <extra_id_99>

        Args:
            mask_indices (np.ndarray[bool]): binary mask of tokens to mask out

        Returns:
            np.ndarray[int]: array of sentinel ids at starting point of spans to be masked out
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (self.tokenizer.vocab_size - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids: np.ndarray[int], sentinel_ids: np.ndarray[int]) -> np.ndarray[int]:
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        Sequence length after masking is deterministic: all masked sequences will have the same length.

        Args:
            input_ids (np.ndarray[int]): sequence of input token ids
            sentinel_ids (np.ndarray[int]): array of sentinel ids at starting point of spans to be masked out

        Returns:
            np.ndarray[int]: input ids with masked out spans replaced by single sentinel token
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.tokenizer.eos_token_id)], axis=-1, dtype=np.int32
        )
        return input_ids

    def random_spans_noise_mask(self, length: int) -> np.ndarray[bool]:
        """Generate a random mask to apply to input.

        Args:
            length (int): length of the noise mask

        Returns:
            np.ndarray: binary noise mask of tokens to mask out
        """
        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))
        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def compute_input_and_target_lengths(
    inputs_length: int, noise_density: float, mean_noise_span_length: float
) -> tuple[int, int]:
    """This function is a copy of random_spans_helper
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>
    This is a helper function that is currently unused by the pipeline,
    but can be used to compute optimal settings for config files.
    Because the span masking collator replaces masked spans with a single token and therefore shortens them,
    we can increase the length of the original token sequence passed to the collator.


    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: an integer - length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length
