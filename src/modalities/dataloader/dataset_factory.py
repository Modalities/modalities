from pathlib import Path
from typing import List, Optional, Tuple, Union

from pydantic import FilePath
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from modalities.dataloader.dataset import (
    DummyDataset,
    DummySampleConfig,
    ImageTransformConfig,
    MemMapDataset,
    PackedMemMapDatasetContinuous,
    PackedMemMapDatasetMegatron,
    WebDataset,
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
    @staticmethod
    def get_dummy_dataset(num_samples: int, sample_definition: Tuple[DummySampleConfig]) -> DummyDataset:
        dataset = DummyDataset(num_samples=num_samples, sample_definition=sample_definition)
        return dataset

    @staticmethod
    def get_mem_map_dataset(
        raw_data_path: Path,
        block_size: int,
        tokenizer: PreTrainedTokenizer,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ) -> MemMapDataset:
        dataset = MemMapDataset(
            raw_data_path=raw_data_path,
            block_size=block_size,
            tokenizer=tokenizer,
            sample_key=sample_key,
            index_path=index_path,
            jq_pattern=jq_pattern,
        )
        return dataset

    @staticmethod
    def get_packed_mem_map_dataset_continuous(
        raw_data_path: Path, block_size: int, sample_key: str
    ) -> PackedMemMapDatasetContinuous:
        dataset = PackedMemMapDatasetContinuous(
            raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key
        )
        return dataset

    @staticmethod
    def get_packed_mem_map_dataset_megatron(
        raw_data_path: Path, block_size: int, sample_key: str
    ) -> PackedMemMapDatasetMegatron:
        dataset = PackedMemMapDatasetMegatron(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)
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

    @staticmethod
    def get_web_dataset(
        urls: Union[List[str], str],
        source_image_key: str,
        image_key: str,
        source_text_key: str,
        text_key: str,
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        num_samples: int,
        image_transform_config: Optional[ImageTransformConfig] = None,
        shardshuffle: Optional[int] = None,
        repeat: bool = False,
        resample: bool = False,
        shuffle: int = 0,
    ) -> WebDataset:
        dataset = WebDataset(
            urls=urls,
            source_image_key=source_image_key,
            image_key=image_key,
            source_text_key=source_text_key,
            text_key=text_key,
            tokenizer=tokenizer,
            block_size=block_size,
            num_samples=num_samples,
            image_transform_config=image_transform_config,
            shardshuffle=shardshuffle,
            repeat=repeat,
            resample=resample,
            shuffle=shuffle,
        )
        return dataset
