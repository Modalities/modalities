import math
from typing import List, Optional

from modalities.nn.weight_init.weight_init import (
    ModulewiseNormalInitialization,
    NamedParameterwiseNormalInitialization,
    WeightInitializationIF,
    WeightInitializerWrapper,
)


class WeightInitializationFactory:
    @staticmethod
    def get_weight_initializer_wrapper(weight_initializers: List[WeightInitializationIF]) -> WeightInitializationIF:
        initializer_wrapper = WeightInitializerWrapper(weight_initializers)
        return initializer_wrapper

    @staticmethod
    def get_plain_weight_initialization(
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
    def get_named_parameterwise_normal_initialization(
        mean: float, std: float, parameter_name_suffixes: List[str]
    ) -> WeightInitializationIF:
        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=std, parameter_name_suffixes=parameter_name_suffixes
        )
        return initialization

    @staticmethod
    def get_scaled_weight_initialization(
        mean: float, plain_std: float, number_of_layers: int, parameter_name_suffixes: List[str]
    ) -> WeightInitializationIF:
        """Implementation of scaled weight initialization.
        For the scaled initialization of the residual projections (i.e., c_proj), as per the GPT-2 paper,
        we need to set parameter_name_suffixes to ["c_proj.weight"].

        Args:
            mean (float): Mean of the normal distribution
            plain_std (float): Standard deviation of the normal distribution used to initialize the other weights
            number_of_layers (int): Number of layers in the model which we use to downscale plain_std with
            parameter_name_suffixes (List[str]): List of parameter name suffixes to which the initialization
                should be applied

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        scaled_std = plain_std / math.sqrt(2 * number_of_layers)

        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=scaled_std, parameter_name_suffixes=parameter_name_suffixes
        )
        return initialization

    def get_scaled_embed_initialization(
        self, mean: float, parameter_name_suffixes: List[str] = None
    ) -> WeightInitializationIF:
        """Implementation of scaled weight initialization for embeddings, see https://arxiv.org/abs/2312.16903
        For the GPT-2 implementation we need to set parameter_name_suffixes to ["wte.weight", "wpe.weight"].
        We fix the standard deviation to 0.4.

        Args:
            mean (float): Mean of the normal distribution
            parameter_name_suffixes (List[str], optional): List of parameter name suffixes to which the initialization
                should be applied Defaults to None.

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        if parameter_name_suffixes is None:
            parameter_name_suffixes = ["wte.weight", "wpe.weight"]
        std = 0.4
        initialization = NamedParameterwiseNormalInitialization(
            mean=mean, std=std, parameter_name_suffixes=parameter_name_suffixes
        )
        return initialization
