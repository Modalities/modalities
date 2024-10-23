from pathlib import Path

import torch
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.batch import DatasetBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.models.gpt2.collator import CollateFnIF
from modalities.running_env.cuda_env import CudaEnv


class CustomGPT2LLMCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str
    custom_attribute: str


class CustomGPT2LLMCollateFn(CollateFnIF):
    def __init__(self, sample_key: str, target_key: str, custom_attribute: str):
        self.sample_key = sample_key
        self.target_key = target_key
        self.custom_attribute = custom_attribute

    def __call__(self, batch: list[list[int]]) -> DatasetBatch:
        sample_tensor = torch.tensor(batch)
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}
        return DatasetBatch(targets=targets, samples=samples)


def main():
    # load and parse the config file
    cwd = Path(__file__).parent
    config_file_path = cwd / Path("config_lorem_ipsum.yaml")
    # instantiate the Main entrypoint of modalities by passing in the config path
    modalities_main = Main(config_path=config_file_path)

    # add the custom component to modalities
    modalities_main.add_custom_component(
        component_key="collate_fn",
        variant_key="custom_gpt_2_llm_collator",
        custom_component=CustomGPT2LLMCollateFn,
        custom_config=CustomGPT2LLMCollateFnConfig,
    )
    # run the experiment
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = modalities_main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        modalities_main.run(components)


if __name__ == "__main__":
    main()
