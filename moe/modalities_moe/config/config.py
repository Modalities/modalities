from typing import Any

from pydantic import BaseModel

from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticPytorchModuleOrListType


class MoECrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str
    model: Any
    tag: str = "MoECrossEntropyLoss"

    class Config:
        arbitrary_types_allowed = True


class EPWrappedModelConfig(BaseModel):
    model: PydanticPytorchModuleOrListType
    block_names: list[str]
    device_mesh: PydanticDeviceMeshIFType
    ep_mesh_dim_name: str | None = None
