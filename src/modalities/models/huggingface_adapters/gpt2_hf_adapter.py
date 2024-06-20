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


class GPT2HuggingFaceAdapterConfig(HuggingFaceAdapterConfig):
    model_type = "modalities_gpt2"


class GPT2HuggingFaceModelAdapter(HuggingFaceModelAdapter):
    config_class = GPT2HuggingFaceAdapterConfig

    def __init__(self, config: HuggingFaceAdapterConfig, model: Optional[nn.Module] = None, *inputs, **kwargs):
        super().__init__(config)
        raise NotImplementedError


class MambaHuggingFaceModelAdapterConfig(HuggingFaceModelAdapterConfig):
    raise NotImplementedError
