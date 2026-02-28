import torch

from modalities.batch import DatasetBatch
from modalities.dataloader.collate_fns.collate_if import CollateFnIF


class GPT2LLMCollateFn(CollateFnIF):
    """GPT2LLMCollateFn class to define a collate function for GPT2 language model."""

    def __init__(
        self,
        sample_key: str,
        target_key: str,
        sub_seq_lengths_key: str | None = None,
        eos_token_id: int | None = None,
        padding_token_id: int | None = None,
    ):
        """
        Initializes the Collator object.
        If the eos token ID and the sub_seq_lengths_key are provided,
        a list[list[int]] representing the sub-sequence lengths will be created.

        Args:
            sample_key (str): The key for accessing the sample data.
            target_key (str): The key for accessing the target data.
            sub_seq_lengths_key (str | None): The key for accessing the sub-sequence lengths.
            eos_token_id (int | None): The end-of-sequence token ID.
            padding_token_id (int | None): The padding token ID.
        """
        self.sample_key = sample_key
        self.target_key = target_key
        self.sub_seq_lengths_key = sub_seq_lengths_key
        self.eos_token_id = eos_token_id
        self.padding_token_id = padding_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
        """
        Process a batch of data.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing tensors.

        Returns:
            DatasetBatch: A processed batch of data where sample and target sequences are created.

        """

        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}
        if self.sub_seq_lengths_key is not None:
            # Determine sub sequence lengths by finding the eos tokens in each sequence in the batch.
            sub_seq_lengths = self._compute_sub_sequence_lengths_for_each_sequence(samples[self.sample_key])
            samples[self.sub_seq_lengths_key] = sub_seq_lengths
        return DatasetBatch(targets=targets, samples=samples)

    def _compute_sub_sequence_lengths_for_each_sequence(self, sample_tensor: torch.Tensor) -> list[list[int]]:
        sub_seq_lengths_in_batch = []
        for batch_seq in sample_tensor:
            eos_positions = (batch_seq == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) == 0:
                assert (
                    self.padding_token_id is None or batch_seq[0] != self.padding_token_id
                ), "Sequence starts with padding token"
                sub_seq_lengths_in_batch.append([len(batch_seq)])
            else:
                lens_in_seq = self._compute_subsequence_length_in_sequence(batch_seq, eos_positions)
                sub_seq_lengths_in_batch.append(lens_in_seq)
        return sub_seq_lengths_in_batch

    def _compute_subsequence_length_in_sequence(self, seq: torch.Tensor, eos_positions: torch.Tensor) -> list[int]:
        # If the last sequence is cut, i.e. does not end on an eos token,
        # it should also be included unless the padding token is set and
        # the last sequence is just padding.
        last_eos_pos = eos_positions[-1].item()
        if self._has_cutoff_final_sequence(seq, last_eos_pos):
            eos_positions = torch.cat([eos_positions, eos_positions.new_tensor([len(seq) - 1])])
        # Compute length of each subsequence and add to lengths list.
        sub_seq_lengths = []
        prev_pos = 0
        for pos in eos_positions:
            sub_seq_lengths.append(pos.item() - prev_pos + 1)
            prev_pos = pos.item() + 1
        return sub_seq_lengths

    def _has_cutoff_final_sequence(self, seq: torch.Tensor, last_eos_pos: int) -> bool:
        # Assumption: If the first token of the last sequence is padding, so is the rest.
        return last_eos_pos < len(seq) - 1 and (
            self.padding_token_id is None or seq[last_eos_pos + 1] != self.padding_token_id
        )
