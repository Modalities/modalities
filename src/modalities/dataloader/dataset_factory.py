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
        raw_data_path: Path, sequence_length: int, sample_key: str, reuse_last_target: Optional[bool] = True
    ) -> PackedMemMapDatasetContinuous:
        """
        Initializes a Dataset object. In case `reuse_last_target` is True,
        we reuse the last target token as the first one for the next sample. If `reuse_last_target` is False,
        we don't reuse the last target in the next sample but never have the the first token of a sample as the target.

        Args:
            raw_data_path (Path): The path to the raw data.
            sequence_length (int): The length of each sequence.
            sample_key (str): The key to access the sample data.
            reuse_last_target (Optional[bool], optional): Whether to reuse the last target. Defaults to True.

        Returns:
            PackedMemMapDatasetContinuous: The created dataset object.
        """
        dataset = PackedMemMapDatasetContinuous(
            raw_data_path=raw_data_path,
            block_size=sequence_length + 1,
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
