from typing import Any, Dict

from pydantic import BaseModel


def convert_base_model_config_to_dict(config: BaseModel) -> Dict[Any, Any]:
    """ "Converts non-recursively a Pydantic BaseModel to a dictionary."""
    return {key: getattr(config, key) for key in config.model_dump().keys()}
