from typing import Annotated

from pydantic import BaseModel, Field


class FQNsPerStageGeneratorConfig(BaseModel):  # TODO duplicate
    pass


class GPT2LLMStagesGeneratorConfig(BaseModel):
    num_model_layers: Annotated[int, Field(strict=True, ge=1)]
    input_layer_equivalence: Annotated[int, Field(strict=True, ge=1)] = 1
    output_layer_equivalence: Annotated[int, Field(strict=True, ge=1)] = 1
