# The layer-spec (declarative builder) pattern is adapted from NVIDIA's Megatron-LM
# (megatron/core/transformer/spec_utils.py::ModuleSpec and
# megatron/core/models/hybrid/hybrid_layer_specs.py).
# Copyright (c) 2023-2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.

"""Layer specifications: configurable builders for the sublayers of a hybrid model.

A hybrid model contains dozens of layers of a handful of types. Modalities' component factory
instantiates every config node exactly once, so injecting *instantiated* layer modules would make
all layers of a type share one set of weights. Layer specs solve this the same way Megatron-LM's
``ModuleSpec`` does: the registry produces a declarative builder, and the model calls
:meth:`NemotronLayerSpecIF.build` once per layer position to get a fresh module.

Every network component (mixer, attention, experts, router, norms) is therefore fully configurable
from YAML while still yielding independent parameters per layer.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator

from modalities.models.components.mamba2.mamba2_mixer import Mamba2Mixer, SSDBackend
from modalities.models.components.moe.experts import ExpertsBackend, GroupedExperts
from modalities.models.components.moe.moe import MoE
from modalities.models.components.moe.router import RouterScoreFunction, TopKRouter
from modalities.models.components.norms import NormWrapperConfig
from modalities.models.nemotron.layer_pattern import LayerSymbol
from modalities.models.nemotron.nemotron_attention import NemotronAttentionImplementation, NemotronSelfAttention
from modalities.models.nemotron.nemotron_layers import (
    Mamba2Layer,
    NemotronAttentionLayer,
    NemotronMLPLayer,
    NemotronMoELayer,
)
from modalities.models.nemotron.nemotron_mlp import SquaredReLUMLP

PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class NemotronLayerSpecIF(ABC):
    """Interface for a builder that produces one residual sublayer per model position."""

    @property
    @abstractmethod
    def symbol(self) -> LayerSymbol:
        """
        The layer pattern symbol this spec is responsible for.

        Returns:
            LayerSymbol: The symbol, e.g. ``LayerSymbol.MAMBA`` for ``"M"``.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, layer_idx: int) -> nn.Module:
        """
        Builds a fresh layer module with its own parameters.

        Args:
            layer_idx (int): The index of the layer within the model. Passed through for specs
                that want depth-dependent behaviour; the current specs ignore it.

        Returns:
            nn.Module: A newly constructed layer.
        """
        raise NotImplementedError


class Mamba2LayerSpecConfig(BaseModel):
    """
    Configuration of a Mamba-2 layer.

    Attributes:
        n_embd (int): The model dimension.
        mamba_n_heads (int): Number of SSM heads.
        mamba_head_dim (int): Dimension of a single SSM head.
        mamba_state_dim (int): SSM state dimension.
        mamba_n_groups (int): Number of B/C groups; must divide ``mamba_n_heads``.
        d_conv (int): Kernel size of the causal depthwise convolution.
        chunk_size (int): Chunk length of the selective scan.
        ssd_backend (SSDBackend): Which scan implementation to use.
        norm_config (NormWrapperConfig): The pre-normalization of the layer.
        mixer_norm_eps (float): Epsilon of the mixer-internal gated RMS norm.
        bias (bool): Whether the input/output projections use a bias.
        conv_bias (bool): Whether the depthwise convolution uses a bias.
    """

    n_embd: PositiveInt
    mamba_n_heads: PositiveInt
    mamba_head_dim: PositiveInt
    mamba_state_dim: PositiveInt
    mamba_n_groups: PositiveInt
    d_conv: PositiveInt = 4
    chunk_size: PositiveInt = 128
    ssd_backend: SSDBackend = SSDBackend.NATIVE
    norm_config: NormWrapperConfig
    mixer_norm_eps: Annotated[float, Field(strict=True, gt=0)] = 1e-5
    bias: bool = False
    conv_bias: bool = True

    @model_validator(mode="after")
    def _validate_head_group_divisibility(self) -> "Mamba2LayerSpecConfig":
        if self.mamba_n_heads % self.mamba_n_groups != 0:
            raise ValueError(
                f"mamba_n_heads ({self.mamba_n_heads}) must be divisible by " f"mamba_n_groups ({self.mamba_n_groups})."
            )
        return self


class Mamba2LayerSpec(NemotronLayerSpecIF):
    """Builds :class:`Mamba2Layer` instances from a :class:`Mamba2LayerSpecConfig`."""

    def __init__(self, **config):
        """
        Initializes the spec.

        Args:
            **config: The fields of :class:`Mamba2LayerSpecConfig`.
        """
        self.config = Mamba2LayerSpecConfig(**config)

    @property
    def symbol(self) -> LayerSymbol:
        return LayerSymbol.MAMBA

    def build(self, layer_idx: int) -> Mamba2Layer:
        config = self.config
        mixer = Mamba2Mixer(
            n_embd=config.n_embd,
            n_heads=config.mamba_n_heads,
            head_dim=config.mamba_head_dim,
            state_dim=config.mamba_state_dim,
            n_groups=config.mamba_n_groups,
            d_conv=config.d_conv,
            chunk_size=config.chunk_size,
            ssd_backend=config.ssd_backend,
            norm_eps=config.mixer_norm_eps,
            bias=config.bias,
            conv_bias=config.conv_bias,
        )
        return Mamba2Layer(norm=config.norm_config.build(), mixer=mixer)


class NemotronAttentionLayerSpecConfig(BaseModel):
    """
    Configuration of a grouped-query self-attention layer.

    Attributes:
        n_embd (int): The model dimension.
        n_head_q (int): Number of query heads.
        n_head_kv (int): Number of key/value heads; must divide ``n_head_q``.
        head_dim (int): Dimension of a single attention head, independent of ``n_embd``.
        attention_implementation (NemotronAttentionImplementation): Which kernel to use.
        norm_config (NormWrapperConfig): The pre-normalization of the layer.
        bias (bool): Whether the projections use a bias.
        dropout (float): Attention dropout probability.
    """

    n_embd: PositiveInt
    n_head_q: PositiveInt
    n_head_kv: PositiveInt
    head_dim: PositiveInt
    attention_implementation: NemotronAttentionImplementation = NemotronAttentionImplementation.PYTORCH_FLASH
    norm_config: NormWrapperConfig
    bias: bool = False
    dropout: Annotated[float, Field(strict=True, ge=0.0, lt=1.0)] = 0.0

    @model_validator(mode="after")
    def _validate_head_divisibility(self) -> "NemotronAttentionLayerSpecConfig":
        if self.n_head_q % self.n_head_kv != 0:
            raise ValueError(f"n_head_q ({self.n_head_q}) must be divisible by n_head_kv ({self.n_head_kv}).")
        return self


class NemotronAttentionLayerSpec(NemotronLayerSpecIF):
    """Builds :class:`NemotronAttentionLayer` instances."""

    def __init__(self, **config):
        """
        Initializes the spec.

        Args:
            **config: The fields of :class:`NemotronAttentionLayerSpecConfig`.
        """
        self.config = NemotronAttentionLayerSpecConfig(**config)

    @property
    def symbol(self) -> LayerSymbol:
        return LayerSymbol.ATTENTION

    def build(self, layer_idx: int) -> NemotronAttentionLayer:
        config = self.config
        attn = NemotronSelfAttention(
            n_embd=config.n_embd,
            n_head_q=config.n_head_q,
            n_head_kv=config.n_head_kv,
            head_dim=config.head_dim,
            attention_implementation=config.attention_implementation,
            bias=config.bias,
            dropout=config.dropout,
        )
        return NemotronAttentionLayer(norm=config.norm_config.build(), attn=attn)


class NemotronMoELayerSpecConfig(BaseModel):
    """
    Configuration of a mixture-of-experts layer.

    Attributes:
        n_embd (int): The model dimension.
        num_experts (int): Number of routed experts.
        moe_ffn_hidden (int): Hidden dimension of a single routed expert.
        top_k (int): Number of experts each token is routed to.
        score_function (RouterScoreFunction): Router score function.
        route_scale (float): Constant factor applied to the routing weights.
        use_expert_bias (bool): Whether to maintain the auxiliary-loss-free balancing bias.
        router_dtype (str): Dtype for router score computation, ``"float32"`` or ``"bfloat16"``.
        num_shared_experts (int): Number of always-on shared experts. Realized as a single MLP of
            ``num_shared_experts * shared_expert_ffn_hidden_per_expert`` hidden units, matching the
            reference implementation. Zero disables shared experts.
        shared_expert_ffn_hidden_per_expert (int | None): Hidden dimension of one shared expert.
            Defaults to ``moe_ffn_hidden``.
        aux_loss_coeff (float): Coefficient of the sequence-level load-balancing loss.
        experts_backend (ExpertsBackend): Which grouped matmul implementation to use.
        norm_config (NormWrapperConfig): The pre-normalization of the layer.
        bias (bool): Whether the shared expert projections use a bias.
    """

    n_embd: PositiveInt
    num_experts: PositiveInt
    moe_ffn_hidden: PositiveInt
    top_k: PositiveInt
    score_function: RouterScoreFunction = RouterScoreFunction.SIGMOID
    route_scale: Annotated[float, Field(strict=True, gt=0.0)] = 1.0
    use_expert_bias: bool = True
    router_dtype: str = "float32"
    num_shared_experts: Annotated[int, Field(strict=True, ge=0)] = 0
    shared_expert_ffn_hidden_per_expert: Optional[PositiveInt] = None
    aux_loss_coeff: Annotated[float, Field(strict=True, ge=0.0)] = 0.0
    experts_backend: ExpertsBackend = ExpertsBackend.GROUPED_MM
    norm_config: NormWrapperConfig
    bias: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "NemotronMoELayerSpecConfig":
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must not exceed num_experts ({self.num_experts}).")
        if self.router_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"router_dtype must be 'float32' or 'bfloat16', got '{self.router_dtype}'.")
        return self

    @property
    def shared_expert_ffn_hidden(self) -> int:
        """
        The total hidden dimension of the fused shared expert MLP.

        Returns:
            int: ``num_shared_experts * shared_expert_ffn_hidden_per_expert``, or 0 if disabled.
        """
        if self.num_shared_experts == 0:
            return 0
        per_expert = self.shared_expert_ffn_hidden_per_expert or self.moe_ffn_hidden
        return self.num_shared_experts * per_expert


class NemotronMoELayerSpec(NemotronLayerSpecIF):
    """Builds :class:`NemotronMoELayer` instances."""

    def __init__(self, **config):
        """
        Initializes the spec.

        Args:
            **config: The fields of :class:`NemotronMoELayerSpecConfig`.
        """
        self.config = NemotronMoELayerSpecConfig(**config)

    @property
    def symbol(self) -> LayerSymbol:
        return LayerSymbol.MOE

    def build(self, layer_idx: int) -> NemotronMoELayer:
        config = self.config
        router = TopKRouter(
            n_embd=config.n_embd,
            num_experts=config.num_experts,
            top_k=config.top_k,
            score_function=config.score_function,
            route_scale=config.route_scale,
            use_expert_bias=config.use_expert_bias,
            router_dtype=getattr(torch, config.router_dtype),
        )
        experts = GroupedExperts(
            n_embd=config.n_embd,
            ffn_hidden=config.moe_ffn_hidden,
            num_experts=config.num_experts,
            backend=config.experts_backend,
        )
        shared_experts = (
            SquaredReLUMLP(n_embd=config.n_embd, ffn_hidden=config.shared_expert_ffn_hidden, bias=config.bias)
            if config.num_shared_experts > 0
            else None
        )
        moe = MoE(
            router=router,
            experts=experts,
            shared_experts=shared_experts,
            aux_loss_coeff=config.aux_loss_coeff,
        )
        return NemotronMoELayer(norm=config.norm_config.build(), moe=moe)


class NemotronMLPLayerSpecConfig(BaseModel):
    """
    Configuration of a dense squared-ReLU feed-forward layer.

    Attributes:
        n_embd (int): The model dimension.
        ffn_hidden (int): The hidden dimension.
        norm_config (NormWrapperConfig): The pre-normalization of the layer.
        bias (bool): Whether the projections use a bias.
    """

    n_embd: PositiveInt
    ffn_hidden: PositiveInt
    norm_config: NormWrapperConfig
    bias: bool = False


class NemotronMLPLayerSpec(NemotronLayerSpecIF):
    """Builds :class:`NemotronMLPLayer` instances."""

    def __init__(self, **config):
        """
        Initializes the spec.

        Args:
            **config: The fields of :class:`NemotronMLPLayerSpecConfig`.
        """
        self.config = NemotronMLPLayerSpecConfig(**config)

    @property
    def symbol(self) -> LayerSymbol:
        return LayerSymbol.MLP

    def build(self, layer_idx: int) -> NemotronMLPLayer:
        config = self.config
        mlp = SquaredReLUMLP(n_embd=config.n_embd, ffn_hidden=config.ffn_hidden, bias=config.bias)
        return NemotronMLPLayer(norm=config.norm_config.build(), mlp=mlp)
