import math
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, model_validator

from modalities.config.pydanctic_if_types import PydanticWeightInitializationIFType
from modalities.nn.weight_init.weight_init import (
    ModulewiseNormalInitialization,
    NamedParameterwiseNormalInitialization,
    WeightInitializationIF,
    WeightInitializerWrapper,
)


class WeightInitializerWrapperConfig(BaseModel):
    weight_initializers: List[PydanticWeightInitializationIFType]


class PlainWeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    hidden_dim: Optional[int] = None

    @model_validator(mode="after")
    def check_std_and_hidden_dim(self):
        if self.std == "auto" and self.hidden_dim is None:
            raise ValueError("hidden_dim must be specified when std is 'auto'")
        return self


class NamedParameterwiseNormalInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class ScaledWeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    plain_std: Annotated[float, Field(strict=True, ge=0.0)]
    num_layers: Annotated[int, Field(strict=True, gt=0)]
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class ScaledEmbedInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class LowLevelInitializationFactory:
    @staticmethod
    def get_weight_initializer_wrapper(weight_initializers: List[WeightInitializationIF]) -> WeightInitializationIF:
        initializer_wrapper = WeightInitializerWrapper(weight_initializers)
        return initializer_wrapper

    @staticmethod
    def get_plain_initialization(
        mean: float, std: float | str, hidden_dim: Optional[int] = None
    ) -> WeightInitializationIF:
        """Initializes the weights of a model by sampling from a normal distribution.
        NOTE: This class supports the initialization of nn.Linear and nn.Embedding layers.
        For other layer types, the initialization must be subclassed and extended

        Args:
            mean (float): mean of the normal distribution
            std (float): standard deviation of the normal distribution
            hidden_dim (Optional[int], optional): hidden dimension of the attention layer. Defaults to None.
        """

        # auto: choose std automatically
        if std == "auto":
            if hidden_dim is None:
                raise ValueError("ERROR! weight_init.std = auto not implemented")

            std = math.sqrt(2 / (5 * hidden_dim))

        initialization = ModulewiseNormalInitialization(mean=mean, std=std)
        return initialization

    @staticmethod
    def get_scaled_initialization(
        mean: float, plain_std: float, num_layers: int, parameter_name_suffixes: List[str]
    ) -> WeightInitializationIF:
        """Implementation of scaled weight initialization.
        For the scaled initialization of the residual projections (i.e., c_proj), as per the GPT-2 paper,
        we need to set parameter_name_suffixes to ["c_proj.weight"].

        Args:
            mean (float): Mean of the normal distribution
            plain_std (float): Standard deviation of the normal distribution used to initialize the other weights
            num_layers (int): Number of layers in the model which we use to downscale plain_std with
            parameter_name_suffixes (List[str]): List of parameter name suffixes to which the initialization
                should be applied

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        scaled_std = plain_std / math.sqrt(2 * num_layers)

        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=scaled_std, parameter_name_suffixes=parameter_name_suffixes
        )
        return initialization

    @staticmethod
    def get_scaled_embed_initialization(mean: float, parameter_name_suffixes: List[str]) -> WeightInitializationIF:
        """Implementation of scaled weight initialization for embeddings, see https://arxiv.org/abs/2312.16903
        We fix the standard deviation to sqrt(0.4).

        Args:
            mean (float): Mean of the normal distribution
            parameter_name_suffixes (List[str], optional): List of parameter name suffixes to which the initialization
                should be applied Defaults to None.

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        std = math.sqrt(0.4)
        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=std, parameter_name_suffixes=parameter_name_suffixes
        )
        return initialization
