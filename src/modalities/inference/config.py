from typing import Annotated, Optional

import torch.nn as nn
from pydantic import BaseModel, FilePath

from modalities.config.config import CudaEnvSettings, PydanticThirdPartyTypeIF, PydanticTokenizerIFType
from modalities.config.lookup_enum import LookupEnum
from modalities.utils.inference_component import InferenceComponent

PydanticPytorchModuleType = Annotated[nn.Module, PydanticThirdPartyTypeIF(nn.Module)]


class DeviceMode(LookupEnum):
    CPU = "cpu"
    SINGLE_GPU = "single-gpu"
    MULTI_GPU = "multi-gpu"


class InferenceSettings(BaseModel):
    model_path: FilePath
    max_new_tokens: int
    cuda_env: CudaEnvSettings
    eod_token: str


class InferenceComponentsModel(BaseModel):
    inference_component: InferenceComponent
    tokenizer: PydanticTokenizerIFType
    settings: InferenceSettings


class InferenceComponentConfig(BaseModel):
    model: PydanticPytorchModuleType
    device_mode: DeviceMode
    gpu_id: Optional[int] = None
