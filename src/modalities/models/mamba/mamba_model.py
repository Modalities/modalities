# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
import sys
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from modalities.models.mamba.mamba_block import Block, MambaBlock
from modalities.models.mamba.mamba_config import MixerModelConfig, MambaBlockConfig
from modalities.models.model import NNModel
from transformers import PreTrainedTokenizer

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
        d_model: int,
        ssm_cfg: dict,
        norm_epsilon: float,
        rms_norm: bool,
        residual_in_fp32: bool,
        fused_add_norm: bool,
        layer_idx: int,
        device: str,
        dtype: str,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(MambaBlock, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(d_model=d_model, mixer_cls=mixer_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm,
                  residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            norm_epsilon: float,
            rms_norm: bool,
            initializer_cfg: dict,
            fused_add_norm: bool,
            residual_in_fp32: bool,
            device: str,
            dtype: str,
            mamba_block_config: MambaBlockConfig,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    ssm_cfg=mamba_block_config.dict(),
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids: torch.Tensor, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaLLM(NNModel):

    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            rms_norm: bool,
            residual_in_fp32: bool,
            fused_add_norm: bool,
            pad_vocab_size_multiple: int,
            tie_embeddings: bool,
            prediction_key: str,
            sample_key: str,
            seed: int,
            dtype: str,
            initializer_cfg: dict,
            num_last_tokens: int,
            inference_params: dict,
            mixer_model_config: MixerModelConfig,
    ):
        super().__init__(seed=seed)

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        self.prediction_key = prediction_key
        self.sample_key = sample_key
        self.dtype = dtype
        self.initializer_cfg = initializer_cfg
        self.mixer_model_config = mixer_model_config

        # todo: How to pass these variables in the forward method?
        self.inference_params = inference_params
        self.num_last_tokens = num_last_tokens

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=self.d_model,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            rms_norm=self.rms_norm,
            initializer_cfg=self.initializer_cfg,
            fused_add_norm=self.fused_add_norm,
            residual_in_fp32=self.residual_in_fp32,
            dtype=self.dtype,
            norm_epsilon=self.mixer_model_config.norm_epsilon,
            device=self.mixer_model_config.device,
            mamba_block_config=self.mixer_model_config.mamba_block_config
        )
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=self.dtype)
        self.apply(
            partial(
                _init_weights,
                n_layer=self.n_layer,
                **initializer_cfg,
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """

        hidden_states = self.backbone(inputs[self.sample_key], inference_params=self.inference_params)
        if self.num_last_tokens > 0:
            hidden_states = hidden_states[:, -self.num_last_tokens:]
        lm_logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        return {self.prediction_key: lm_logits}

    def generate(
            self,
            tokenizer: PreTrainedTokenizer,
            context: str,
            max_new_tokens: int,
            temperature: float = 1.0,
    ):
        if not context:
            raise ValueError("Context must be not empty")

        in_batch = tokenizer([context])
        in_batch[self.sample_key] = torch.Tensor(in_batch[self.sample_key]).to(torch.int32).to(
            next(self.parameters()).device)

        for _ in range(max_new_tokens):
            logits = self.forward(in_batch)[self.prediction_key]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next_str = tokenizer.decode(idx_next[0])
            if idx_next_str == tokenizer.eos_token:
                print("\n<reached eos token>", end="")
                break
            else:
                print(idx_next_str, end="")
                sys.stdout.flush()
                in_batch[self.sample_key] = torch.cat((in_batch[self.sample_key], idx_next), dim=1)
        print("")
