import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import PydanticPytorchModuleType, load_app_config_dict
from modalities.models.huggingface_adapters.hf_adapter import HFAdapterConfig, HFAdapter
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class CheckpointConversion:

    def __init__(
            self, config_file_path: Path, output_hf_checkpoint_dir: Path,
    ):
        self.output_hf_checkpoint_dir = output_hf_checkpoint_dir
        if not config_file_path.exists():
            raise ValueError(f"Could not find {config_file_path}")

        self.config_dict = load_app_config_dict(config_file_path)
        logging.info(f"Config\n{self.config_dict}")

    def convert_pytorch_to_hf_checkpoint(self):
        model = self._setup_model()
        config = HFAdapterConfig(config=self.config_dict)
        hf_model = HFAdapter(config=config, model=model)
        hf_model.save_pretrained(self.output_hf_checkpoint_dir, safe_serialization=False)
        return hf_model

    def _setup_model(self):
        registry = Registry(COMPONENTS)
        component_factory = ComponentFactory(registry=registry)

        class ModelConfig(BaseModel):
            checkpointed_model: PydanticPytorchModuleType

        components = component_factory.build_components(
            config_dict=self.config_dict, components_model_type=ModelConfig
        )
        return components.checkpointed_model
