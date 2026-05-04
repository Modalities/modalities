from pydantic import BaseModel

from modalities.config.pydantic_if_types import (
    PydanticDatasetBatchGeneratorIFType,
    PydanticLossIFType,
    PydanticOptimizerIFType,
    PydanticPytorchModuleType,
)


class SteppableForwardPassConfig(BaseModel):
    model: PydanticPytorchModuleType
    dataset_batch_generator: PydanticDatasetBatchGeneratorIFType
    loss_fn: PydanticLossIFType | None = None
    optimizer: PydanticOptimizerIFType | None = None
