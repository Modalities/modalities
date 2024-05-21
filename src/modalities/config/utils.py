from typing import Any, Dict

import torch
from pydantic import BaseModel


def convert_base_model_config_to_dict(config: BaseModel) -> Dict[Any, Any]:
    """ "Converts non-recursively a Pydantic BaseModel to a dictionary."""
    return {key: getattr(config, key) for key in config.model_dump().keys()}


def parse_torch_device(device: str | int) -> torch.device:
    if isinstance(device, str) and device != "cpu":
        raise ValueError(f"Invalid device_id: {device}")
    elif isinstance(device, int):
        device_id = f"cuda:{device}"
    else:
        device_id = "cpu"
    device = torch.device(device_id)
    return device
