from typing import Optional

from pydantic import BaseModel, field_validator

from modalities.config.pydanctic_if_types import (
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
    PydanticTokenizerIFType,
)
from modalities.config.utils import parse_torch_device


class TextInferenceComponentConfig(BaseModel):
    model: PydanticPytorchModuleType
    tokenizer: PydanticTokenizerIFType
    prompt_template: str
    sequence_length: int
    temperature: Optional[float] = 1.0
    eod_token: Optional[str] = "<eod>"
    device: PydanticPytorchDeviceType

    @field_validator("device", mode="before")
    def parse_device(cls, device) -> PydanticPytorchDeviceType:
        return parse_torch_device(device)
