from typing import Annotated, Dict

from pydantic import BaseModel, FilePath, field_validator

from modalities.config.config import (
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
    PydanticThirdPartyTypeIF,
    PydanticTokenizerIFType,
)
from modalities.config.lookup_enum import LookupEnum
from modalities.config.utils import parse_torch_device
from modalities.inference.text.inference_component import TextInferenceComponent

# PydanticPytorchModuleType = Annotated[nn.Module, PydanticThirdPartyTypeIF(nn.Module)]

PydanticTextInferenceComponentType = Annotated[TextInferenceComponent, PydanticThirdPartyTypeIF(TextInferenceComponent)]


class DeviceMode(LookupEnum):
    CPU = "cpu"
    SINGLE_GPU = "single-gpu"
    MULTI_GPU = "multi-gpu"


class TextGenerationSettings(BaseModel):
    model_path: FilePath
    context_length: int
    device: PydanticPytorchDeviceType
    referencing_keys: Dict[str, str]

    @field_validator("device", mode="before")
    def parse_device(cls, device) -> PydanticPytorchDeviceType:
        return parse_torch_device(device)


class TextGenerationInstantiationModel(BaseModel):
    text_inference_component: PydanticTextInferenceComponentType
    settings: TextGenerationSettings


class InferenceComponentConfig(BaseModel):
    model: PydanticPytorchModuleType
    tokenizer: PydanticTokenizerIFType
    context_length: int
    eod_token: str
    device: PydanticPytorchDeviceType
    prompt_template: str
    temperature: float

    @field_validator("device", mode="before")
    def parse_device(cls, device) -> PydanticPytorchDeviceType:
        return parse_torch_device(device)
