import logging
from pathlib import Path

from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import PydanticPytorchModuleType, load_app_config_dict
from modalities.models.huggingface.huggingface_adapter import HuggingFaceAdapterConfig, HuggingFaceModel
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class CheckpointConversion:
    def __init__(
            self, checkpoint_dir: Path, config_file_name: str, model_file_name: str, output_hf_checkpoint_dir: Path
    ):
        self.initial_model = None
        self.config_dict = None
        self.checkpoint_dir = checkpoint_dir
        self.config_file_name = config_file_name
        self.model_file_name = model_file_name
        self.output_hf_checkpoint_dir = output_hf_checkpoint_dir
        self.registry = Registry(COMPONENTS)
        self.component_factory = ComponentFactory(registry=self.registry)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        config_file_path = checkpoint_dir / self.config_file_name
        if not config_file_path.exists():
            raise ValueError(f"Could not find {self.config_file_name} in {checkpoint_dir}")

        self.config_dict = load_app_config_dict(config_file_path)
        logging.info(f"Config\n{self.config_dict}")

        self.checkpoint_path = checkpoint_dir / self.model_file_name

    def convert_pytorch_to_hf_checkpoint(self):
        # FIXME: make the conversion entry point configurable from outside:
        # Which HuggingFaceAdapterConfig should be used etxactly currently it is too hard coded:
        # Allow for a custom callable for conversion to HF to be given to the entrypoint
        model = self._setup_model()
        config = HuggingFaceAdapterConfig()
        hf_model = HuggingFaceModel(config=config, model=model)
        hf_model.save_pretrained(self.output_hf_checkpoint_dir, safe_serialization=False)
        return hf_model

    def _setup_model(self):
        class ModelConfig(BaseModel):
            checkpointed_model: PydanticPytorchModuleType

        components = self.component_factory.build_components(
            config_dict=self.config_dict, components_model_type=ModelConfig
        )
        return components.checkpointed_model
