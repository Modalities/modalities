from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, Type, Union

import jq
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchdataSet
from torch.utils.data.dataset import Subset

from llm_gym.gpt2.collator import Tokenizer

from ..dataloader.large_file_lines_reader import LargeFileLinesReader


@dataclasses.dataclass
class DatasetSplit:
    train: Subset
    validation: Subset
    test: Subset


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Union[str, Path]):
        self.raw_data_path = Path(raw_data_path)

    @staticmethod
    def from_path(
        dataset_path: str, target_dataset_cls: Type[Dataset], split_size: Iterable[float] = (0.9, 0.05, 0.05)
    ) -> DatasetSplit:
        presplit_dataset_folder_paths = [Path(dataset_path, split) for split in ["train", "test", "validation"]]

        def get_subset(ds):
            return Subset(dataset=ds, indices=range(len(ds)))

        if all(p.is_dir() for p in presplit_dataset_folder_paths):
            print(f"Found already existing dataset split at {dataset_path}. Will use this one...")

            def init_dataset(path):
                return target_dataset_cls(raw_data_path=path)

            loaded_datasets = map(init_dataset, presplit_dataset_folder_paths)

            loaded_subsets = map(get_subset, loaded_datasets)
            dataset_split = tuple(loaded_subsets)
        else:
            print(f"No existing dataset split found at {dataset_path}. Loading dataset directly and apply split")
            dataset = target_dataset_cls(raw_data_path=dataset_path)
            dataset_split = random_split(dataset, split_size)
        return DatasetSplit(
            train=get_subset(dataset_split[0]),
            validation=get_subset(dataset_split[1]),
            test=get_subset(dataset_split[2]),
        )


class MemMapDataset(Dataset):
    def __init__(self, raw_data_path: Union[str, Path], jq_filter: str = ".text"):
        super().__init__(raw_data_path=raw_data_path)
        self.reader = LargeFileLinesReader(self.raw_data_path, lazy_init=True)
        self.jq_filter = jq.compile(jq_filter)
        self.tokenizer = Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.reader)

    # TODO: tokenizer singleton?
    def __getitem__(self, idx: int) -> str:
        obj = self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(), max_length=1024, padding="max_length", truncation=True
        )
        return obj
