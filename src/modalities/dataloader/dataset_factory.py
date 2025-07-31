import pickle
from pathlib import Path
from typing import Optional

from transformers import PreTrainedTokenizer

from modalities.dataloader.dataset import (
    CombinedDataset,
    Dataset,
    DummyDataset,
    DummySampleConfig,
    MemMapDataset,
    PackedMemMapDatasetContinuous,
    PackedMemMapDatasetMegatron,
)


class DatasetFactory:
    """DatasetFactory for building the different dataset types."""

    @staticmethod
    def get_dummy_dataset(num_samples: int, sample_definition: tuple[DummySampleConfig]) -> DummyDataset:
        """
        Returns a DummyDataset object.

        Args:
            num_samples (int): The number of samples the dataset should generate.
            sample_definition (tuple[DummySampleConfig]): A list of tuples defining the dataset output.
                Each tuple contains the sample key, shape and data type.

        Returns:
            DummyDataset: The generated DummyDataset object.
        """
        dataset = DummyDataset(num_samples=num_samples, sample_definition=sample_definition)
        return dataset

    @staticmethod
    def get_mem_map_dataset(
        raw_data_path: Path,
        tokenizer: PreTrainedTokenizer,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ) -> MemMapDataset:
        """
        Returns a MemMapDataset object.

        Args:
            raw_data_path (Path): The path to the raw data.
            tokenizer (PreTrainedTokenizer): The tokenizer used to tokenize the data.
            sample_key (str): The key used to retrieve the samples from the dataset.
            index_path (Optional[Path], optional): The path to the index file. Defaults to None.
            jq_pattern (str, optional): The pattern used to extract the text from the data. Defaults to ".text".

        Returns:
            MemMapDataset: The MemMapDataset object.

        """
        dataset = MemMapDataset(
            raw_data_path=raw_data_path,
            tokenizer=tokenizer,
            sample_key=sample_key,
            index_path=index_path,
            jq_pattern=jq_pattern,
        )
        return dataset

    @staticmethod
    def get_raw_index(raw_index_path: Path) -> list[tuple[int, int]]:
        with raw_index_path.open("rb") as f:
            index = pickle.load(f)
        return index

    @staticmethod
    def get_packed_mem_map_dataset_continuous(
        raw_data_path: Path, sequence_length: int, sample_key: str, reuse_last_target: bool
    ) -> PackedMemMapDatasetContinuous:
        """
        Initializes a PackedMemMapDatasetContinuous object. If `reuse_last_target` is True,
        the last target token of one sample is reused as the first input token of the next sample,
        creating an overlap of one token between samples (recommended for pre-training).
        If `reuse_last_target` is False, there is no overlap:
        Each sample is a distinct block, and the first token of each sample is never used as a target
        (recommended for instruction tuning).

        Args:
            raw_data_path (Path): The path to the raw data.
            sequence_length (int): The length of each sequence.
            sample_key (str): The key used to retrieve the samples from the dataset.
            reuse_last_target (bool): Whether to reuse the last target.

        Returns:
            PackedMemMapDatasetContinuous: The packed memory-mapped dataset.
        """
        dataset = PackedMemMapDatasetContinuous(
            raw_data_path=raw_data_path,
            # we can increase the block size by 1, as we reuse the last target token
            # if we do not reuse the last target token, we should not increase the block size, as the this would lead to
            # getting samples with increasing offset, e.g.:
            # [0, 1, 2, ..., sequence_length - 1] for the first sample,
            # [1, 2, 3, ..., sequence_length] for the second sample
            # and so on, which is not what we want.
            block_size=(sequence_length + 1) if reuse_last_target else sequence_length,
            sample_key=sample_key,
            reuse_last_target=reuse_last_target,
        )
        return dataset

    @staticmethod
    def get_packed_mem_map_dataset_megatron(
        raw_data_path: Path, sequence_length: int, sample_key: str
    ) -> PackedMemMapDatasetMegatron:
        dataset = PackedMemMapDatasetMegatron(
            raw_data_path=raw_data_path, block_size=sequence_length + 1, sample_key=sample_key
        )
        return dataset

    @staticmethod
    def get_combined_dataset(datasets: list[Dataset]) -> Dataset:
        """Factory method for creating a combined datset .

        Args:
            datasets (list[Dataset]): List of datasets to combine.

        Returns:
            Dataset: CombinedDataset object.
        """
        return CombinedDataset(datasets=datasets)
