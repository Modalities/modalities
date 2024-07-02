import math
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, model_validator

from modalities.nn.weight_init.weight_init import NamedParameterwiseNormalInitialization, WeightInitializationIF


class PlainInitializationConfig(BaseModel):
    mean: float
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    parameter_name_regexes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"
    hidden_dim: Optional[int] = None

    @model_validator(mode="after")
    def check_std_and_hidden_dim(self):
        if self.std == "auto" and self.hidden_dim is None:
            raise ValueError("hidden_dim must be specified when std is 'auto'")
        return self


class ScaledInitializationConfig(BaseModel):
    mean: float
    std: Annotated[float, Field(strict=True, ge=0.0)]
    num_layers: Annotated[int, Field(strict=True, gt=0)]
    parameter_name_regexes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class ScaledEmbedInitializationConfig(BaseModel):
    mean: float
    parameter_name_regexes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class LowLevelInitializationFactory:
    @staticmethod
    def get_plain_initialization(
        mean: float, std: float | str, parameter_name_regexes: List[str], hidden_dim: Optional[int] = None
    ) -> NamedParameterwiseNormalInitialization:
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
            # as per  https://arxiv.org/abs/2312.16903
            std = math.sqrt(2 / (5 * hidden_dim))

        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=std, parameter_name_regexes=parameter_name_regexes
        )
        return initialization

    @staticmethod
    def get_scaled_initialization(
        mean: float, std: float, num_layers: int, parameter_name_regexes: List[str]
    ) -> WeightInitializationIF:
        """Implementation of scaled weight initialization.

        Args:
            mean (float): Mean of the normal distribution
            std (float): Standard deviation of the normal distribution used to initialize the other weights
            num_layers (int): Number of layers in the model which we use to downscale std with
            parameter_name_regexes (List[str]): List of parameter name regexes to which the initialization
                should be applied

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        scaled_std = std / math.sqrt(2 * num_layers)

        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=scaled_std, parameter_name_regexes=parameter_name_regexes
        )
        return initialization

    @staticmethod
    def get_scaled_embed_initialization(mean: float, parameter_name_regexes: List[str]) -> WeightInitializationIF:
        """Implementation of scaled weight initialization for embeddings, see https://arxiv.org/abs/2312.16903
        We fix the standard deviation to sqrt(0.4).

        Args:
            mean (float): Mean of the normal distribution
            parameter_name_regexes (List[str], optional): List of parameter name regexes to which the initialization
                should be applied Defaults to None.

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        std = math.sqrt(0.4)
        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=std, parameter_name_regexes=parameter_name_regexes
        )
        return initialization
