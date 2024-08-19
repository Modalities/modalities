from pathlib import Path
from typing import Optional, Tuple

from pydantic import FilePath
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from modalities.dataloader.dataset import (
    DummyDataset,
    DummySampleConfig,
    MemMapDataset,
    PackedMemMapDatasetContinuous,
    PackedMemMapDatasetMegatron,
)
from modalities.dataloader.open_gptx_dataset.open_gptx_dataset import OpenGPTXMMapDataset


class OpenGPTXDatasetWrapper(Dataset):
    def __init__(self, open_gptx_dataset: OpenGPTXMMapDataset, num_samples: int) -> None:
        super().__init__()
        self.open_gptx_dataset = open_gptx_dataset
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.num_samples > idx:
            return self.open_gptx_dataset.__getitem__(idx)
        else:
            raise ValueError("num_samples <= idx")


class DatasetFactory:
    """DatasetFactory class."""

    @staticmethod
    def get_dummy_dataset(num_samples: int, sample_definition: Tuple[DummySampleConfig]) -> DummyDataset:
        """
        Returns a DummyDataset object.

        Args:
            num_samples (int): The number of samples the dataset should generate.
            sample_definition (Tuple[DummySampleConfig]): A list of tuples defining the dataset output.
                Each tuple contains the sample key, shape and data type.

        Returns:
            DummyDataset: The generated DummyDataset object.
        """
        dataset = DummyDataset(num_samples=num_samples, sample_definition=sample_definition)
        return dataset

    @staticmethod
    def get_mem_map_dataset(
        raw_data_path: Path,
        sequence_length: int,
        tokenizer: PreTrainedTokenizer,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ) -> MemMapDataset:
        """
        Returns a MemMapDataset object.

        Args:
            raw_data_path (Path): The path to the raw data.
            sequence_length (int): The length of each sequence.
            tokenizer (PreTrainedTokenizer): The tokenizer used to tokenize the data.
            sample_key (str): The key used to retrieve the samples from the dataset.
            index_path (Optional[Path], optional): The path to the index file. Defaults to None.
            jq_pattern (str, optional): The pattern used to extract the text from the data. Defaults to ".text".

        Returns:
            MemMapDataset: The MemMapDataset object.

        """
        dataset = MemMapDataset(
            raw_data_path=raw_data_path,
            block_size=sequence_length + 1,
            tokenizer=tokenizer,
            sample_key=sample_key,
            index_path=index_path,
            jq_pattern=jq_pattern,
        )
        return dataset

    @staticmethod
    def get_packed_mem_map_dataset_continuous(
        raw_data_path: Path, sequence_length: int, sample_key: str
    ) -> PackedMemMapDatasetContinuous:
        """
        Returns a PackedMemMapDatasetContinuous object.

        Args:
            raw_data_path (Path): The path to the raw data.
            sequence_length (int): The length of each sequence.
            sample_key (str): The key used to retrieve the samples from the dataset.

        Returns:
            PackedMemMapDatasetContinuous: The packed memory-mapped dataset.

        """
        dataset = PackedMemMapDatasetContinuous(
            raw_data_path=raw_data_path, block_size=sequence_length + 1, sample_key=sample_key
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
    def get_open_gptx_mmap_dataset(
        sample_key: str,
        path: FilePath,
        sequence_len: int,
        num_samples: int,
        seed: int = 47,
    ) -> OpenGPTXMMapDataset:
        # part of open gptx
        dataset = OpenGPTXMMapDataset(
            sample_key=sample_key, path=path, sequence_len=sequence_len, num_samples=num_samples, seed=seed
        )

        # BUG: Sometimes the dataset genereated by the OpenGPTXMMap implementation has too many samples.
        # This is a workaround to fix the dataset to the size, as specified in the config!
        # TODO: Fix the OpenGPTX implementation and get rid of this hack.
        dataset_wrapped = OpenGPTXDatasetWrapper(open_gptx_dataset=dataset, num_samples=num_samples)
        return dataset_wrapped
