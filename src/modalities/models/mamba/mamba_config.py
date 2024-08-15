from typing import Optional

from pydantic import BaseModel


class MambaBlockConfig(BaseModel):
    """
    Configuration class for MambaBlock.

    Attributes:
        d_state (int): The dimension of the state.
        d_conv (int): The dimension of the convolution.
        expand (int): The expansion factor.
        dt_rank (str): -
        dt_min (float): -.
        dt_max (float): -.
        dt_init (str): -.
        dt_scale (float): -.
        dt_init_floor (float): -
        conv_bias (bool): Flag indicating whether to use bias in convolution.
        bias (bool): Flag indicating whether to use bias.
        use_fast_path (bool): -.
    """

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
    """
    Configuration class for the MixerModel.

    Args:
        norm_epsilon (float): The epsilon value for normalization.
        device (Optional[str]): The device to use for the model.
        mamba_block_config (MambaBlockConfig): The configuration for the MambaBlock.

    """

    norm_epsilon: float
    device: Optional[str]
    mamba_block_config: MambaBlockConfig


class MambaLLMConfig(BaseModel):
    """
    Configuration class for MambaLLM model.

    Args:
        d_model (int): The dimensionality of the feature vectors.
        n_layer (int): The number of layers in the model.
        vocab_size (int): The size of the vocabulary.
        rms_norm (bool): Defines, whether to apply root mean square normalization.
        residual_in_fp32 (bool): Defines, whether to use FP32 for residual connections.
        fused_add_norm (bool): Defines, whether to use fused add and normalization function.
        pad_vocab_size_multiple (int): The multiple of vocabulary size to pad to.
        tie_embeddings (bool): Defiesn, whether to tie the input and output embeddings.
        prediction_key (str): The prediction keys.
        sample_key (str): The sample keys.
        seed (Optional[int]): The random seed for reproducibility.
        dtype (Optional[int]): The data type of the model.
        initializer_cfg (dict): The configuration for model initialization.
        num_last_tokens (int): -.
        inference_params (dict): The parameters for inference.
        mixer_model_config (MixerModelConfig): The configuration for the mixer model.
    """

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
