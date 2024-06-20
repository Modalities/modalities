import json
from abc import ABC
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from modalities.models.huggingface_adapters.hf_adapter import HuggingFaceAdapterConfig, HuggingFaceModelAdapter, \
    HuggingFaceModelAdapterConfig
from modalities.models.mamba.mamba_config import MambaBlockConfig, MixerModelConfig
from modalities.models.mamba.mamba_model import MambaLLM


class MambaHuggingFaceAdapterConfig(HuggingFaceAdapterConfig):
    model_type = "modalities_mamba"


class MambaHuggingFaceModelAdapter(HuggingFaceModelAdapter):
    config_class = MambaHuggingFaceAdapterConfig

    def __init__(self, config: HuggingFaceAdapterConfig, model: Optional[nn.Module] = None, *inputs, **kwargs):
        super().__init__(config)
        if not model:
            mamba_llm_config = self.convert_config_to_mamba_llm_config(config)
            self.model: MambaLLM = MambaLLM(**mamba_llm_config)

            self.config = self.convert_config_config_to_pydantic(mamba_llm_config)
        else:
            self.model = model

    def convert_config_to_mamba_llm_config(self, config):
        config.config["model"]["config"]["mixer_model_config"]["mamba_block_config"] = MambaBlockConfig(
            **config.config["model"]["config"]["mixer_model_config"]["mamba_block_config"])
        config.config["model"]["config"]["mixer_model_config"] = MixerModelConfig(
            **config.config["model"]["config"]["mixer_model_config"])
        return config.config["model"]["config"]

    def convert_config_config_to_pydantic(self, config):
        config["is_encoder_decoder"] = False
        return MambaHuggingFaceModelAdapterConfig(**config)


class MambaHuggingFaceModelAdapterConfig(HuggingFaceModelAdapterConfig):
    d_model: int
    n_layer: int
    vocab_size: int
    rms_norm: bool
    residual_in_fp32: bool
    fused_add_norm: bool
    pad_vocab_size_multiple: int
    tie_embeddings: bool
    prediction_key: str
    sample_key: str
    seed: Optional[int]
    dtype: Optional[str]
    initializer_cfg: Dict
    num_last_tokens: int
    inference_params: Dict
    mixer_model_config: MixerModelConfig
    is_encoder_decoder: bool
