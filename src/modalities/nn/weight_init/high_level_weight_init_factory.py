from typing import List, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

from modalities.config.pydanctic_if_types import PydanticWeightInitializationIFType
from modalities.nn.weight_init.low_level_weight_init_factory import LowLevelInitializationFactory
from modalities.nn.weight_init.parameter_name_regex_filters import (
    NAMED_PARAMETER_INIT_GROUPS,
    SupportWeightInitModels,
    WeightInitTypes,
)
from modalities.nn.weight_init.weight_init import WeightInitializationIF, WeightInitializerWrapper


class WeightInitializerWrapperConfig(BaseModel):
    weight_initializers: List[PydanticWeightInitializationIFType]


class ComposedWeightInitializationConfig(BaseModel):
    model_type: SupportWeightInitModels
    weight_init_type: WeightInitTypes

    mean: float
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    hidden_dim: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    num_layers: Optional[Annotated[int, Field(strict=True, gt=0)]] = None

    @model_validator(mode="after")
    def _check_values(self):
        # in case of initialization with "auto", we need to specify the hidden_dim
        if self.std == "auto" and self.hidden_dim is None:
            raise ValueError("hidden_dim must be specified when std is 'auto'")

        # in case of plain initialization the number of layers is not required
        if self.weight_init_type == WeightInitTypes.PLAIN and self.num_layers is not None:
            raise ValueError("num_layers must not be specified when weight_init_type is plain")

        # in case of scaled or scaled_embed we need to specify the number of layers
        if self.weight_init_type in [WeightInitTypes.SCALED, WeightInitTypes.SCALED_EMBED] and self.num_layers is None:
            raise ValueError("num_layers must be specified when weight_init_type is scaled or scaled_embed")

        # in case of scaled or scaled_embed we additionally need to check if the model supports scaled initialisation.
        # (scaled_embed requires scaled initialization to be run before)
        if self.weight_init_type in [WeightInitTypes.SCALED, WeightInitTypes.SCALED_EMBED]:
            scaled_parameter_name_regexes = NAMED_PARAMETER_INIT_GROUPS[self.model_type][WeightInitTypes.SCALED]
            if scaled_parameter_name_regexes is None:
                raise ValueError(
                    f"Model type {self.model_type.value} does not support weight "
                    "init type {self.weight_init_type.value}"
                )

        # in case of scaled_embed we need to check if the model supports it.
        if self.weight_init_type == WeightInitTypes.SCALED_EMBED:
            scaled_embed_parameter_name_regexes = NAMED_PARAMETER_INIT_GROUPS[self.model_type][
                WeightInitTypes.SCALED_EMBED
            ]
            if scaled_embed_parameter_name_regexes is None:
                raise ValueError(
                    f"Model type {self.model_type.value} does not support weight "
                    "init type {self.weight_init_type.value}"
                )

        return self


class HighLevelWeightInitializationFactory:
    @staticmethod
    def get_weight_initializer_wrapper(weight_initializers: List[WeightInitializationIF]) -> WeightInitializationIF:
        initializer_wrapper = WeightInitializerWrapper(weight_initializers)
        return initializer_wrapper

    @staticmethod
    def get_composed_weight_init(
        model_type: SupportWeightInitModels,
        weight_init_type: WeightInitTypes,
        mean: float,
        std: float | str,
        hidden_dim: Optional[int] = None,
        num_layers: int = None,
    ) -> WeightInitializationIF:
        """This initialization allows to intialize a model with plain, scaled or scaled_embed initialization.
        Note that plain initialization is always performed in the beginning. In case of scaled_embed,
        also scaled is being performed before scaled_embed and after plain.

        Args:
            model_type (SupportWeightInitModels): Model type enum referencing the model (e.g., "gpt2")
            weight_init_type (WeightInitTypes): The initialization method we want to perform.
            mean (float): Mean of the normal distribution
            std (float | str): Standard deviation of the plain normal distribution
            hidden_dim (Optional[int], optional): Hidden dimension size of the model (required for plain if std="auto").
                Defaults to None.
            num_layers (int, optional): Number of layers in the model (required for scaled and scaled_embed only).
                Defaults to None.

        Returns:
            WeightInitializationIF: The Weight Initializer performing the initialization as specified.
        """
        weight_initializers = []

        # plain
        plain_parameter_name_regexes = NAMED_PARAMETER_INIT_GROUPS[model_type][WeightInitTypes.PLAIN]
        plain_init = LowLevelInitializationFactory.get_plain_initialization(
            mean=mean, std=std, hidden_dim=hidden_dim, parameter_name_regexes=plain_parameter_name_regexes
        )
        working_std = plain_init.std
        weight_initializers.append(plain_init)

        if weight_init_type in [WeightInitTypes.SCALED, WeightInitTypes.SCALED_EMBED]:
            # scaled
            scaled_parameter_name_regexes = NAMED_PARAMETER_INIT_GROUPS[model_type][WeightInitTypes.SCALED]
            scaled_init = LowLevelInitializationFactory.get_scaled_initialization(
                mean=mean,
                std=working_std,
                num_layers=num_layers,
                parameter_name_regexes=scaled_parameter_name_regexes,
            )
            weight_initializers.append(scaled_init)

        if weight_init_type == WeightInitTypes.SCALED_EMBED:
            # scaled embed
            scaled_embed_parameter_name_regexes = NAMED_PARAMETER_INIT_GROUPS[model_type][WeightInitTypes.SCALED_EMBED]
            scaled_embed_init = LowLevelInitializationFactory.get_scaled_embed_initialization(
                mean=mean, parameter_name_regexes=scaled_embed_parameter_name_regexes
            )
            weight_initializers.append(scaled_embed_init)

        # composition of multiple weight initializers
        init_wrapper = HighLevelWeightInitializationFactory.get_weight_initializer_wrapper(
            weight_initializers=weight_initializers
        )
        return init_wrapper
