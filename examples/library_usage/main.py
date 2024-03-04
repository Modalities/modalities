from pathlib import Path
from typing import Dict, List

import torch
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.batch import DatasetBatch
from modalities.config.config import load_app_config_dict
from modalities.models.gpt2.collator import CollateFnIF


class CustomGPT2LLMCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str
    custom_attribute: str


class CustomGPT2LLMCollateFn(CollateFnIF):
    def __init__(self, sample_key: str, target_key: str, custom_attribute: str):
        self.sample_key = sample_key
        self.target_key = target_key
        self.custom_attribute = custom_attribute

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}

        print(f"Custom attribute: {self.custom_attribute}")

        return DatasetBatch(targets=targets, samples=samples)


def main():
    # load and parse the config file
    config_file_path = Path("config_lorem_ipsum.yaml")
    config_dict = load_app_config_dict(config_file_path)

    # instantiate the Main entrypoint of modalities by passing in the config
    modalities_main = Main(config_dict=config_dict, config_path=config_file_path)

    # add the custom component to modalities
    modalities_main.add_custom_component(
        component_key="collate_fn",
        variant_key="custom_gpt_2_llm_collator",
        custom_component=CustomGPT2LLMCollateFn,
        custom_config=CustomGPT2LLMCollateFnConfig,
    )
    # run the experiment
    modalities_main.run()


if __name__ == "__main__":
    main()
