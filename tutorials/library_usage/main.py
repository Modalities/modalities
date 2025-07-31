import os
from pathlib import Path

import torch
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.batch import DatasetBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.dataloader.collate_fns.collate_if import CollateFnIF
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
        self._num_calls = 0

    @property
    def num_calls(self) -> int:
        return self._num_calls

    def __call__(self, batch: list[list[int]]) -> DatasetBatch:
        sample_tensor = torch.tensor(batch)
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}
        self._num_calls += 1
        return DatasetBatch(targets=targets, samples=samples)


def main():
    # load and parse the config file
    cwd = Path(__file__).parent
    # change to cwd
    os.chdir(cwd)
    config_file_path = cwd / Path("config_lorem_ipsum.yaml")

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
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
        components: TrainingComponentsInstantiationModel = modalities_main.build_components(
            components_model_type=TrainingComponentsInstantiationModel
        )
        modalities_main.run(components)

        collate_fn = components.train_dataloader.collate_fn
        if collate_fn.num_calls < 1:
            raise ValueError("Custom collator was not called during training.")
        print(f"Custom collator was called {collate_fn.num_calls} times during training.")


if __name__ == "__main__":
    main()
