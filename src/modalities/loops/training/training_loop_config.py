from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydanctic_if_types import PydanticGradientClipperIFType, PydanticMessagePublisherIFType


class TrainingLoopConfig(BaseModel):
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    batch_progress_publisher: PydanticMessagePublisherIFType
    step_state_publisher: PydanticMessagePublisherIFType
    gradient_acc_steps: Annotated[float, Field(strict=True, ge=1)]
    gradient_clipper: PydanticGradientClipperIFType
