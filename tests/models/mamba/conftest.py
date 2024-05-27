import torch
from torch import nn

from modalities.models.mamba.mamba_block import MambaBlock, Block
import pytest

from functools import partial
from modalities.models.mamba.mamba_config import MambaBlockConfig, MixerModelConfig
from modalities.models.mamba.mamba_model import MambaLLM, MixerModel


@pytest.fixture()
def batch_size():
    return 2


@pytest.fixture()
def expand():
    return 2


@pytest.fixture(scope="session")
def d_model():
    return 16


@pytest.fixture()
def d_state():
    return 3


@pytest.fixture(scope="session")
def d_conv():
    return 4


@pytest.fixture()
def vocab_size():
    return 1024


@pytest.fixture()
def sequence_length():
    return 64


@pytest.fixture()
def n_layer():
    return 2


@pytest.fixture()
def conv_state(batch_size, expand, d_model, d_conv):
    return torch.rand((batch_size, expand * d_model, d_conv))


@pytest.fixture()
def ssm_state(batch_size, expand, d_model, d_state):
    return torch.rand((batch_size, expand * d_model, d_state))


@pytest.fixture()
def layer_idx():
    return 0


@pytest.fixture()
def dt_rank():
    return "auto"


@pytest.fixture()
def dt_min():
    return 0.001


@pytest.fixture()
def dt_max():
    return 0.1


@pytest.fixture()
def dt_init():
    return "random"


@pytest.fixture()
def dt_scale():
    return 1.0


@pytest.fixture()
def dt_init_floor():
    return 1e-4


@pytest.fixture()
def conv_bias():
    return True


@pytest.fixture()
def bias():
    return False


@pytest.fixture()
def use_fast_path():
    return True


@pytest.fixture()
def device():
    return None


@pytest.fixture()
def dtype():
    return None


@pytest.fixture()
def ssm_cfg(mamba_block_config):
    return mamba_block_config.model_dump()


@pytest.fixture()
def mamba_block_config(d_state,
                       d_conv,
                       expand,
                       dt_rank,
                       dt_min,
                       dt_max,
                       dt_init,
                       dt_scale,
                       dt_init_floor,
                       conv_bias,
                       bias,
                       use_fast_path):
    return MambaBlockConfig(d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            dt_rank=dt_rank,
                            dt_min=dt_min,
                            dt_max=dt_max,
                            dt_init=dt_init,
                            dt_scale=dt_scale,
                            dt_init_floor=dt_init_floor,
                            conv_bias=conv_bias,
                            bias=bias,
                            use_fast_path=use_fast_path)


@pytest.fixture()
def mixer_model(d_model, n_layer, vocab_size, norm_epsilon, rms_norm, initializer_cfg, fused_add_norm, residual_in_fp32,
                device, dtype, mamba_block_config):
    return MixerModel(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size, norm_epsilon=norm_epsilon,
                      rms_norm=rms_norm, initializer_cfg=initializer_cfg, fused_add_norm=fused_add_norm,
                      residual_in_fp32=residual_in_fp32, device=device, dtype=dtype,
                      mamba_block_config=mamba_block_config)


@pytest.fixture()
def factory_kwargs(device, dtype):
    return {"device": device, "dtype": dtype}


@pytest.fixture()
def norm_epsilon():
    return 1e-5


@pytest.fixture()
def rms_norm():
    return False


@pytest.fixture()
def initializer_cfg():
    return {}


@pytest.fixture()
def mixer_cls(layer_idx, factory_kwargs, ssm_cfg):
    return partial(MambaBlock, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)


@pytest.fixture()
def mamba_block(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max, dt_init, dt_init_floor, dt_scale,
                bias, conv_bias, use_fast_path, layer_idx, dtype, device):
    return MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank,
                      dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
                      bias=bias, conv_bias=conv_bias, use_fast_path=use_fast_path, layer_idx=layer_idx, dtype=dtype,
                      device=device)


@pytest.fixture()
def norm_cls(d_model, norm_epsilon, factory_kwargs):
    return partial(
        nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
    )


@pytest.fixture()
def fused_add_norm():
    return True


@pytest.fixture()
def residual_in_fp32():
    return True


@pytest.fixture()
def block(d_model, mixer_cls, norm_cls, fused_add_norm, residual_in_fp32):
    return Block(d_model=d_model,
                 mixer_cls=mixer_cls,
                 norm_cls=norm_cls,
                 fused_add_norm=fused_add_norm,
                 residual_in_fp32=residual_in_fp32)


@pytest.fixture()
def hidden_states(d_model,
                  batch_size, ):
    return torch.randn(batch_size, 1, d_model)


@pytest.fixture()
def prediction_key():
    return "logits"


@pytest.fixture()
def sample_key():
    return "input_ids"


@pytest.fixture()
def seed():
    return 42


@pytest.fixture()
def mixer_model_config(norm_epsilon, device, mamba_block_config):
    return MixerModelConfig(norm_epsilon=norm_epsilon, device=device, mamba_block_config=mamba_block_config)


@pytest.fixture()
def mamba_llm(d_model, n_layer, vocab_size, rms_norm, residual_in_fp32, fused_add_norm, prediction_key, sample_key,
              seed, dtype, initializer_cfg, mixer_model_config):
    return MambaLLM(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size, rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, pad_vocab_size_multiple=1,
                    tie_embeddings=False, prediction_key=prediction_key, sample_key=sample_key, seed=seed, dtype=dtype,
                    initializer_cfg=initializer_cfg, num_last_tokens=0, inference_params={},
                    mixer_model_config=mixer_model_config)


@pytest.fixture()
def linear_layer():
    return nn.Linear(in_features=16, out_features=24)


@pytest.fixture()
def embedding_layer(vocab_size, d_model):
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
