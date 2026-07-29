"""Factory for the Nemotron hybrid Mamba-Transformer model."""

from typing import Optional

import torch

from modalities.models.components.norms import NormWrapperConfig
from modalities.models.nemotron.nemotron_layer_specs import NemotronLayerSpecIF
from modalities.models.nemotron.nemotron_model import NemotronLLM


class NemotronModelFactory:
    """Factory class creating Nemotron models."""

    @staticmethod
    def get_nemotron_model(
        sample_key: str,
        prediction_key: str,
        sequence_length: int,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        layer_pattern: str,
        layer_specs: dict[str, NemotronLayerSpecIF],
        lm_head_norm_config: NormWrapperConfig | dict,
        use_weight_tying: bool = False,
        aux_loss_key: Optional[str] = None,
        use_meta_device: Optional[bool] = False,
        enforce_tensor_core_alignment: bool = True,
    ) -> NemotronLLM:
        """
        Creates a NemotronLLM, optionally on the meta device.

        Args:
            sample_key (str): Key under which the input token ids are found.
            prediction_key (str): Key under which the logits are stored.
            sequence_length (int): Maximum supported sequence length.
            vocab_size (int): Vocabulary size.
            n_embd (int): Model dimension.
            n_layer (int): Number of layers.
            layer_pattern (str): One symbol per layer.
            layer_specs (dict[str, NemotronLayerSpecIF]): Builders keyed by layer pattern symbol.
            lm_head_norm_config (NormWrapperConfig | dict): Normalization before the language model
                head. A plain dict is accepted so that the factory can also be called directly,
                outside the component factory that would normally validate it.
            use_weight_tying (bool): Whether to tie the embedding and the output projection.
            aux_loss_key (str | None): Key under which the summed MoE auxiliary loss is exposed.
            use_meta_device (bool): Whether to build the model on the meta device. Materialization
                and initialization then happen in ``ModelFactory.get_weight_initialized_model``.
            enforce_tensor_core_alignment (bool): Validated in the config; accepted here so that the
                config and the factory signature stay in sync.

        Raises:
            ValueError: If weight tying is combined with meta device initialization.

        Returns:
            NemotronLLM: The constructed model.
        """
        del enforce_tensor_core_alignment  # validated by NemotronLLMConfig
        if not isinstance(lm_head_norm_config, NormWrapperConfig):
            lm_head_norm_config = NormWrapperConfig.model_validate(lm_head_norm_config)
        config = dict(
            sample_key=sample_key,
            prediction_key=prediction_key,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            layer_pattern=layer_pattern,
            layer_specs=layer_specs,
            lm_head_norm_config=lm_head_norm_config,
            use_weight_tying=use_weight_tying,
            aux_loss_key=aux_loss_key,
        )
        if use_meta_device and use_weight_tying:
            raise ValueError(
                "Weight tying is not supported on the meta device. Set either use_meta_device=False "
                "or use_weight_tying=False. See https://github.com/Modalities/modalities/issues/357"
            )
        if use_meta_device:
            with torch.device("meta"):
                return NemotronLLM(**config)
        return NemotronLLM(**config)
