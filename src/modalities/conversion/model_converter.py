import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelConverter:
    """Converts a Modalities checkpoint to HF format by running a user-provided command template.

    The command_template is a shell command string with placeholders:
        {checkpoint_path} - resolved from last_checkpoint_info.json
        {output_dir}      - <checkpoint_path>/hf_checkpoint
        {modalities_config} - path to the modalities YAML config (if available)

    Example command_templates:
        "python convert_gpt2.py {modalities_config} {output_dir}"
        "python convert_adaptive_gpt.py {checkpoint_path} {output_dir} --modalities_config {modalities_config}"
    """

    def __init__(
        self,
        command_template: str,
        checkpoint_dir: Path,
        global_rank: int,
        eval_interval: int,
    ) -> None:
        self.command_template = command_template
        self.checkpoint_dir = Path(checkpoint_dir)
        self.global_rank = global_rank
        self.eval_interval = eval_interval

    def convert(self, num_train_steps_done: int) -> None:
        """Run the conversion command if the current step matches the eval interval.

        Args:
            num_train_steps_done: Number of training steps completed so far.
        """
        if num_train_steps_done == 0 or num_train_steps_done % self.eval_interval != 0:
            return
        if self.global_rank != 0:
            return

        checkpoint_path = self._get_latest_checkpoint_path()
        if checkpoint_path is None:
            logger.warning("No checkpoint info found, skipping conversion.")
            return

        output_dir = checkpoint_path / "hf_checkpoint"
        if output_dir.exists():
            logger.info(f"HF checkpoint already exists at {output_dir}, skipping conversion.")
            return

        cmd = self.command_template.format(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            modalities_config=str(self._find_config_in_checkpoint(checkpoint_path)),
        )

        logger.info(f"Running model conversion: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Conversion complete. HF checkpoint saved to {output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Model conversion failed with return code {e.returncode}: {e}")

    def _get_latest_checkpoint_path(self) -> Path | None:
        """Read last_checkpoint_info.json to find the latest checkpoint path."""
        info_file = self.checkpoint_dir / "last_checkpoint_info.json"
        if not info_file.exists():
            return None

        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        # DCP checkpoints use "checkpoint_folder_path", FSDP1 uses "model_checkpoint_path"
        checkpoint_path_str = info.get("checkpoint_folder_path") or info.get("model_checkpoint_path")
        if checkpoint_path_str is None:
            return None

        path = Path(checkpoint_path_str)
        # For FSDP1, model_checkpoint_path points to the .bin file; we want the parent directory
        if path.is_file():
            path = path.parent
        return path

    @staticmethod
    def _find_config_in_checkpoint(checkpoint_path: Path) -> Path | None:
        """Look for a YAML config file inside or next to the checkpoint directory."""
        for search_dir in [checkpoint_path, checkpoint_path.parent]:
            for f in search_dir.iterdir():
                if f.suffix in (".yaml", ".yml") and not f.name.endswith(".resolved"):
                    return f
        return Path("")
