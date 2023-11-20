from __future__ import annotations

import dataclasses
from typing import Iterable, List, Type, Union

import jq
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchdataSet
from torch.utils.data.dataset import Subset
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

from ..dataloader.large_file_lines_reader import BaseReader


@dataclasses.dataclass
class DatasetSplit:
    train: Subset
    validation: Subset
    test: Subset


class Dataset(TorchdataSet):
    def __init__(self, reader: BaseReader):
        self.reader = reader

    @staticmethod
    def from_reader(
        reader: Union[BaseReader, List[BaseReader]],
        target_dataset_cls: Type[Dataset],
        split_size: Iterable[float] = (0.9, 0.05, 0.05),
        **kwargs,
    ) -> DatasetSplit:
        def get_subset(ds):
            return Subset(dataset=ds, indices=range(len(ds)))

        if isinstance(reader, list):
            if len(reader) != 3:
                raise ValueError(f"Lenght of list of readers needs to be 3 (is {len(reader)}).")
            print("Different readers for train, test and eval were passed. Will use this split...")

            def init_dataset(reader, **kwargs):
                return target_dataset_cls(reader=reader, **kwargs)

            dataset_split = [init_dataset(r, **kwargs) for r in reader]
        else:
            print("No existing dataset split passed. Loading dataset directly and applying split...")
            dataset = target_dataset_cls(reader=reader, **kwargs)
            dataset_split = random_split(dataset, split_size)
        return DatasetSplit(
            train=get_subset(dataset_split[0]),
            validation=get_subset(dataset_split[1]),
            test=get_subset(dataset_split[2]),
        )


class MemMapDataset(Dataset):
    def __init__(
        self,
        reader: BaseReader,
        tokenizer: PreTrainedTokenizerFast = GPT2TokenizerFast(tokenizer_file="./data/tokenizer/tokenizer.json"),
        jq_pattern: str = ".text",
    ):
        super().__init__(reader=reader)
        self.jq_filter = jq.compile(jq_pattern)
        # TODO: tokenizer from tiktoken if it is faster?
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.reader)

    # TODO: tokenizer singleton?
    def __getitem__(self, idx: int) -> str:
        obj = self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(), max_length=1024, padding="max_length", truncation=True
        )
        return obj
