from pydantic import BaseModel

from modalities.config.pydanctic_if_types import PydanticMessagePublisherIFType


class EvaluationLoopConfig(BaseModel):
    batch_progress_publisher: PydanticMessagePublisherIFType
    step_state_publisher: PydanticMessagePublisherIFType
