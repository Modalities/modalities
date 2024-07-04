from pathlib import Path

from modalities.config.config import load_app_config_dict
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapterConfig, HFModelAdapter


class CheckpointConversion:

    def __init__(
        self,
        config_file_path: Path,
        output_hf_checkpoint_dir: Path,
    ):
        self.output_hf_checkpoint_dir = output_hf_checkpoint_dir
        if not config_file_path.exists():
            raise ValueError(f"Could not find {config_file_path}")

        self.config_dict = load_app_config_dict(config_file_path)

    def convert_pytorch_to_hf_checkpoint(self) -> HFModelAdapter:
        config = HFModelAdapterConfig(config=self.config_dict)
        hf_model = HFModelAdapter(config=config)
        hf_model.save_pretrained(self.output_hf_checkpoint_dir, safe_serialization=False)
        return hf_model
