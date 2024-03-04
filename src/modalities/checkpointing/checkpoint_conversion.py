import logging
from pathlib import Path
from typing import Union

from pydantic import BaseModel
import torch

from modalities.config.config import PydanticModelIFType, GPT2HuggingFaceAdapterConfig, load_app_config_dict
from modalities.models.gpt2.gpt2_model import GPT2LLMConfig
from modalities.checkpointing.checkpointing_execution import PytorchToDiscCheckpointing
from modalities.models.gpt2.huggingface_model import HuggingFaceModel

from src.modalities.config.component_factory import ComponentFactory
from src.modalities.registry.components import COMPONENTS
from src.modalities.registry.registry import Registry


class CheckpointConversion:
    def __init__(
        self,
        checkpoint_dir,
        config_file_name,
        model_file_name,
        output_hf_checkpoint_dir
    ):
        self.initial_model = None
        self.config_dict = None
        self.components = None
        self.checkpoint_dir = checkpoint_dir
        self.config_file_name = config_file_name
        self.model_file_name = model_file_name
        self.output_hf_checkpoint_dir = output_hf_checkpoint_dir
        self.registry = Registry(COMPONENTS)
        self.component_factory = ComponentFactory(registry=self.registry)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        input_pytorch_config_file_path = checkpoint_dir / self.config_file_name
        if not input_pytorch_config_file_path.exists():
            raise ValueError(f"Could not find {self.config_file_name} in {checkpoint_dir}")

        # TODO resolve config hierarchically
        self.config_dict = load_app_config_dict(input_pytorch_config_file_path)
        logging.info(f"Config\n{self.config_dict}")

        self.input_pytorch_checkpoint_path = checkpoint_dir / self.model_file_name

    def convert_pytorch_to_hf_checkpoint(self):
        # FIXME: make the conversion entry point configurable from outside:
        # Which HuggingFaceAdapterConfig should be used etxactly currently it is too hard coded:
        # Allow for a custom callable for conversion to HF to be given to the entrypoint

        model = self._setup_model()
        model = self._get_model_from_checkpoint(model)
        model_config = GPT2HuggingFaceAdapterConfig(GPT2LLMConfig(**self.config_dict['model']['config']))
        self._convert_checkpoint(model, model_config)

    def _setup_model(self):
        class ModelConfig(BaseModel):
            model: PydanticModelIFType

        self.components = self.component_factory.build_components(
            config_dict=self.config_dict,
            components_model_type=ModelConfig)

        return self.components.model

    def _get_model_from_checkpoint(self, model: torch.nn.Module):
        if torch.distributed.is_initialized():
            raise NotImplementedError("Checkpoint conversion is only implemented for non-distributed environments")
        checkpointing = PytorchToDiscCheckpointing()
        if not self.input_pytorch_checkpoint_path.exists():
            raise ValueError(f"Could not find model.bin in {self.input_pytorch_checkpoint_path}")
        model = checkpointing.load_model_checkpoint(model, self.input_pytorch_checkpoint_path)
        return model

    def _convert_checkpoint(self, model: torch.nn.Module, model_config: BaseModel):
        hugging_face_model = HuggingFaceModel(config=model_config, model=model)
        hugging_face_model.save_pretrained(self.output_hf_checkpoint_dir, safe_serialization=False)
