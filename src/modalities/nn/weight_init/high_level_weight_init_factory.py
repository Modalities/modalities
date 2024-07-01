from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

from modalities.nn.weight_init.low_level_weight_init_factory import LowLevelInitializationFactory
from modalities.nn.weight_init.weight_init import WeightInitializationIF


class WeightInitTypes(Enum):
    plain = "plain"
    scaled = "scaled"
    scaled_embed = "scaled_embed"


class SupportWeightInitModels(Enum):
    gpt2 = "gpt2"
    coca = "coca"


INITIALIZATION_GROUPS = {
    SupportWeightInitModels.gpt2: {
        # as per https://arxiv.org/abs/2312.16903
        WeightInitTypes.scaled: ["c_proj.weight"],
        WeightInitTypes.scaled_embed: ["wte.weight", "wpe.weight"],
    },
    SupportWeightInitModels.coca: {
        WeightInitTypes.scaled: None,
        WeightInitTypes.scaled_embed: None,
    },
}


class ComposedWeightInitializationConfig(BaseModel):
    model_type: SupportWeightInitModels
    weight_init_type: WeightInitTypes

    mean: float
    plain_std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    hidden_dim: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    num_layers: Optional[Annotated[int, Field(strict=True, gt=0)]] = None

    @model_validator(mode="after")
    def check_values(self):
        # in case of the plain initialization with "auto", we need to specify the hidden_dim
        if self.plain_std == "auto" and self.hidden_dim is None:
            raise ValueError("hidden_dim must be specified when plain_std is 'auto'")

        # in case of plain initialization the number of layers is not rquired
        if self.weight_init_type == WeightInitTypes.plain and self.num_layers is not None:
            raise ValueError("num_layers must not be specified when weight_init_type is plain")

        # in case of scaled or scaled_embed we need to specify the number of layers
        if self.weight_init_type in [WeightInitTypes.scaled, WeightInitTypes.scaled_embed] and self.num_layers is None:
            raise ValueError("num_layers must be specified when weight_init_type is scaled or scaled_embed")

        # in case of scaled or scaled_embed we additionally need to check if the model supports scaled initialisation.
        # (scaled_embed requires scaled initialization to be run before)
        if self.weight_init_type in [WeightInitTypes.scaled, WeightInitTypes.scaled_embed]:
            scaled_parameter_name_suffixes = INITIALIZATION_GROUPS[self.model_type][WeightInitTypes.scaled]
            if scaled_parameter_name_suffixes is None:
                raise ValueError(
                    f"Model type {self.model_type.value} does not support weight "
                    "init type {self.weight_init_type.value}"
                )

        # in case of scaled_embed we need to check if the model supports it.
        if self.weight_init_type == WeightInitTypes.scaled_embed:
            scaled_embed_parameter_name_suffixes = INITIALIZATION_GROUPS[self.model_type][WeightInitTypes.scaled_embed]
            if scaled_embed_parameter_name_suffixes is None:
                raise ValueError(
                    f"Model type {self.model_type.value} does not support weight "
                    "init type {self.weight_init_type.value}"
                )

        return self


class HighLevelWeightInitializationFactory:
    @staticmethod
    def get_composed_weight_init(
        model_type: SupportWeightInitModels,
        weight_init_type: WeightInitTypes,
        mean: float,
        plain_std: float | str,
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
            plain_std (float | str): Standard deviation of the plain normal distribution
            hidden_dim (Optional[int], optional): Hidden dimension size of the model (required for plain if std="auto").
                Defaults to None.
            num_layers (int, optional): Number of layers in the model (required for scaled and scaled_embed only).
                Defaults to None.

        Returns:
            WeightInitializationIF: The Weight Initializer performing the initialization as specified.
        """
        weight_initializers = []

        # plain
        plain_init = LowLevelInitializationFactory.get_plain_initialization(
            mean=mean, std=plain_std, hidden_dim=hidden_dim
        )
        weight_initializers.append(plain_init)

        if weight_init_type in [WeightInitTypes.scaled, WeightInitTypes.scaled_embed]:
            # scaled
            scaled_parameter_name_suffixes = INITIALIZATION_GROUPS[model_type][WeightInitTypes.scaled]
            scaled_init = LowLevelInitializationFactory.get_scaled_initialization(
                mean=mean,
                plain_std=plain_std,
                num_layers=num_layers,
                parameter_name_suffixes=scaled_parameter_name_suffixes,
            )
            weight_initializers.append(scaled_init)

        if weight_init_type == WeightInitTypes.scaled_embed:
            # scaled embed
            scaled_embed_parameter_name_suffixes = INITIALIZATION_GROUPS[model_type][WeightInitTypes.scaled_embed]
            scaled_embed_init = LowLevelInitializationFactory.get_scaled_embed_initialization(
                mean=mean, parameter_name_suffixes=scaled_embed_parameter_name_suffixes
            )
            weight_initializers.append(scaled_embed_init)

        # composition of multiple weight initializers
        init_wrapper = LowLevelInitializationFactory.get_weight_initializer_wrapper(
            weight_initializers=weight_initializers
        )
        return init_wrapper
