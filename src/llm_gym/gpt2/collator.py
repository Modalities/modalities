from dataclasses import field
from llm_gym.exceptions import DatasetNotFoundError
from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast
from typing import List, Dict, Union
import torch
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets import load_from_disk
import os
from llm_gym.batch import DatasetBatch


class GPT2LLMCollator:

    def __init__(self, target_publication_key: str, tokenizer_file_path: str, pad_to_multiple_of: int = 8):
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        self.target_publication_key = target_publication_key
        tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_file_path)  # "trained_wiki_tokenizer/tokenizer.json"
        tokenizer.pad_token = tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                             mlm=False,
                                                             pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, batch: List[torch.Tensor]) -> DatasetBatch:
        """
        :param batch: batch format [no_samples, height, width, channels]
        :return:
        """
        collated_batch = self.data_collator(batch)
        samples = {"input_ids": collated_batch["input_ids"],
                   "attention_mask": collated_batch["attention_mask"]}
        targets = {self.target_publication_key: collated_batch["labels"],
                   "attention_mask": collated_batch["attention_mask"]}
        return DatasetBatch(targets=targets, samples=samples)


class LMWikiBookCorpusDatasetFactory:

    @staticmethod
    def construct(dataset_folder_path: str) -> Dict[str, Union[Dataset, DatasetDict]]:
        dataset_dict = {}
        if dataset_folder_path is None:
            raise DatasetNotFoundError("Dataset path not specified")
        splits = ["train", "test", "validation"]
        for split_name in splits:
            wiki_dataset = load_from_disk(os.path.join(dataset_folder_path, split_name))
            dataset_dict[split_name] = wiki_dataset
        return dataset_dict
