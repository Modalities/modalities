from typing import Annotated, Dict, Optional

from pydantic import BaseModel, FilePath, field_validator

from modalities.config.config import (
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
    PydanticThirdPartyTypeIF,
    PydanticTokenizerIFType,
)
from modalities.config.utils import parse_torch_device
from modalities.inference.text.inference_component import TextInferenceComponent

PydanticTextInferenceComponentType = Annotated[TextInferenceComponent, PydanticThirdPartyTypeIF(TextInferenceComponent)]


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


class TextInferenceComponentConfig(BaseModel):
    model: PydanticPytorchModuleType
    tokenizer: PydanticTokenizerIFType
    prompt_template: str
    context_length: int
    temperature: Optional[float] = 1.0
    eod_token: Optional[str] = "<eod>"
