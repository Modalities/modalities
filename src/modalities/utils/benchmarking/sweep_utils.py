import hashlib
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, FilePath

from modalities.utils.logger_utils import get_logger

logger = get_logger(name=__file__)


class SweepConfig(BaseModel):
    sweep: Dict[str, Any]
    paired: List[List[str]] = []


class SweepGenerator:
    def __init__(
        self,
        sweep_config: SweepConfig,
        output_dir: Path,
    ) -> None:
        """
        Initialize the SBatchArrayJobGenerator with cluster configuration, script configuration,
        sweep configuration, and script template.
        """
        self.sweep_config: SweepConfig = sweep_config
        self.sweep_output_dir_path = output_dir
        self.sweep_output_dir_path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _load_yaml_file(file_path: FilePath) -> Dict[str, Any]:
        """
        Load a YAML file and return its content as a dictionary.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def _get_config_hash(config: Dict[str, Any]) -> str:
        """
        Generate a hash for the given configuration dictionary while preserving the key order.
        """
        # Serialize the dictionary to a YAML string while keeping the key order
        config_str = yaml.dump(config, sort_keys=False, default_flow_style=False)

        # Compute the hash using SHA256
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]
        return config_hash

    @staticmethod
    def generate_sweep_configs(sweep_config_path: Path, output_dir: Path, world_sizes: list[int]) -> None:
        sweep_field = "sweep"
        sweep_config = SweepGenerator._load_yaml_file(sweep_config_path)
        sweep_part = sweep_config.get(sweep_field, {})
        config_part = deepcopy(sweep_config)
        config_part.pop(sweep_field, None)

        sweep_combinations = SweepGenerator._generate_nested_combinations(sweep_part)

        if len(sweep_combinations) == 1:
            logger.warning("Sweep combinations are less than 2. This is not a sweep, but a single configuration. ")
        sweep_hash = SweepGenerator._get_config_hash(sweep_config)
        ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        for sweep_combination in sweep_combinations:
            config_hash = SweepGenerator._get_config_hash(sweep_combination)
            sweep_combination = {sweep_field: sweep_combination, **config_part}
            for world_size in world_sizes:
                config_file_path = (
                    output_dir / f"{ts}_{sweep_hash}" / f"{world_size}" / f"{config_hash}_{ts}" / sweep_config_path.name
                )
                config_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file_path, "w", encoding="utf-8") as file:
                    yaml.dump(sweep_combination, file, sort_keys=False)

    @staticmethod
    def _generate_nested_combinations(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
        def expand(sweep_dict):
            if isinstance(sweep_dict, dict):
                keys, values = zip(*((k, expand(v)) for k, v in sweep_dict.items()))
                return [dict(zip(keys, combo)) for combo in product(*values)]
            elif isinstance(sweep_dict, list):
                return sweep_dict
            else:
                return [sweep_dict]

        return expand(sweep)
