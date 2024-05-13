from typing import Optional

from pydantic import BaseModel


class MambaBlockConfig(BaseModel):
    d_state: int
    d_conv: int
    expand: int
    dt_rank: str
    dt_min: float
    dt_max: float
    dt_init: str
    dt_scale: float
    dt_init_floor: float
    conv_bias: bool
    bias: bool
    use_fast_path: bool


class MixerModelConfig(BaseModel):
    norm_epsilon: float
    device: Optional[str]
    mamba_block_config: MambaBlockConfig


class MambaLLMConfig(BaseModel):
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
    dtype: Optional[int]
    initializer_cfg: dict
    num_last_tokens: int
    inference_params: dict
    mixer_model_config: MixerModelConfig
