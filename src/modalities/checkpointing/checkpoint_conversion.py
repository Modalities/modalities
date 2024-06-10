import logging
from pathlib import Path

import torch
from pydantic import BaseModel

# from modalities.checkpointing.checkpointing_execution import PytorchToDiscCheckpointing
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import PydanticPytorchModuleType, load_app_config_dict
# from modalities.models.gpt2.gpt2_model import GPT2HuggingFaceAdapterConfig, GPT2LLMConfig
# from modalities.models.gpt2.huggingface_model import HuggingFaceModel
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
        breakpoint()
        model = self._setup_model()
        # model = self._get_model_from_checkpoint(model)
        # model_config = GPT2HuggingFaceAdapterConfig(GPT2LLMConfig(**self.config_dict["model"]["config"]))
        # self._convert_checkpoint(model, model_config)

    def _setup_model(self):
        class ModelConfig(BaseModel):
            model: PydanticPytorchModuleType

        components = self.component_factory.build_components(
            config_dict=self.config_dict, components_model_type=ModelConfig
        )

        return components.model

    # def _get_model_from_checkpoint(self, model: torch.nn.Module):
    #     if torch.distributed.is_initialized():
    #         raise NotImplementedError("Checkpoint conversion is only implemented for non-distributed environments")
    #     checkpointing = PytorchToDiscCheckpointing()
    #     if not self.checkpoint_path.exists():
    #         raise ValueError(f"Could not find model.bin in {self.checkpoint_path}")
    #     model = checkpointing.load_model_checkpoint(model, self.checkpoint_path)
    #     return model

    def _convert_checkpoint(self, pytorch_model: torch.nn.Module, hf_adapter_model_config: HuggingFaceAdapterConfig):
        hugging_face_model = HuggingFaceModel(config=hf_adapter_model_config, model=pytorch_model)
        hugging_face_model.save_pretrained(self.output_hf_checkpoint_dir, safe_serialization=False)
