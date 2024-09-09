from pathlib import Path

from modalities.config.config import load_app_config_dict
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter, HFModelAdapterConfig


class CheckpointConversion:
    """Class to convert a PyTorch checkpoint to a Hugging Face checkpoint."""

    def __init__(
        self,
        config_file_path: Path,
        output_hf_checkpoint_dir: Path,
    ):
        """
        Initializes the CheckpointConversion object.

        Args:
            config_file_path (Path): The path to the configuration file containing the pytorch model configuration.
            output_hf_checkpoint_dir (Path): The path to the output Hugging Face checkpoint directory.

        Raises:
            ValueError: If the config_file_path does not exist.

        """
        self.output_hf_checkpoint_dir = output_hf_checkpoint_dir
        if not config_file_path.exists():
            raise ValueError(f"Could not find {config_file_path}.")

        self.config_dict = load_app_config_dict(config_file_path)

    def convert_pytorch_to_hf_checkpoint(self, prediction_key: str) -> HFModelAdapter:
        """
        Converts a PyTorch checkpoint to a Hugging Face checkpoint.

        Args:
            prediction_key (str): The prediction key to be used in the HFModelAdapter.

        Returns:
            HFModelAdapter: The converted Hugging Face model adapter.

        """
        config = HFModelAdapterConfig(config=self.config_dict)
        hf_model = HFModelAdapter(config=config, prediction_key=prediction_key, load_checkpoint=True)
        hf_model.save_pretrained(self.output_hf_checkpoint_dir, safe_serialization=False)
        return hf_model
