#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    """
    Parent Class for Lora Embedding and Lora Linear Layer with main functionalities.

    Args:
        r (int): Rank of the low-rank approximation.
        lora_alpha (int): Scaling factor for the low-rank approximation.
        lora_dropout (float): Dropout rate for LoRA.
        merge_weights (bool): Flag to merge weights during evaluation.
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    Converted Embedding Layer.

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of each embedding.
        r (int, optional): Rank of the low-rank approximation. Default is 0.
        lora_alpha (int, optional): Scaling factor for the low-rank approximation. Default is 1.
        merge_weights (bool, optional): Flag to merge weights during evaluation. Default is True.

    Raises:
        ValueError: If r <= 0.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        # __super__ calls to parents
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        else:
            raise ValueError("r should be > 0")
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize A and B and reset the Embedding Layer parameters.
        Initialize A the same way as the default for nn.Linear and B to zero.
        """

        # init weights as normal distribution and set all padding_idx tokens to zero vectors
        nn.Embedding.reset_parameters(self)
        # we need this since the super.__init__() calls are also calling this function
        if hasattr(self, "lora_A"):
            nn.init.normal_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def train(self, training_mode: bool = True):
        """
        Put the Layer into train/eval mode.

        Args:
            training_mode (bool): If True, set to training mode; else, set to evaluation mode.
        """
        # put embedding layer to training (True) or evaluation (False) mode
        nn.Embedding.train(self, mode=training_mode)
        if training_mode:
            # during training
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged and revert the merge if it has already happened
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            # during e.g. evaluation
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Forward pass for LoRAEmbedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                input=x,
                weight=self.lora_A.transpose(0, 1),
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRALinear(nn.Linear, LoRALayer):
    """
    Converted Linear Layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        r (int, optional): Rank of the low-rank approximation. Default is 0.
        lora_alpha (int, optional): Scaling factor for the low-rank approximation. Default is 1.
        lora_dropout (float, optional): Dropout rate for LoRA. Default is 0.0.
        fan_in_fan_out (bool, optional): If True, the layer stores weight like (fan_in, fan_out). Default is False.
        merge_weights (bool, optional): Flag to merge weights during evaluation. Default is True.

    Raises:
        ValueError: If r <= 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        else:
            raise ValueError("r should be > 0")
        self.reset_parameters()
        if fan_in_fan_out:
            # transpose the layer weights
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        """
        Initialize A and B and reset the Embedding Layer parameters.
        Initialize A the same way as the default for nn.Linear and B to zero.
        """
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, training_mode: bool = True):
        """
        Put the Layer into train/eval mode.

        Args:
            training_mode (bool): If True, set to training mode; else, set to evaluation mode.
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode=training_mode)
        self.to(device=self.weight.device)
        if training_mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Compute forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if not self.merged:
            # If A and B are not merged to the weights, we do y = linear(x) + dropout(x) * A * B * scaling
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            # A and B have already been added to weights
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRAMergedLinearLayer(nn.Linear, LoRALayer):
    """
    If q, k, and v matrix are merged in one linear layer, use this class to convert the layer.
    # This ...
        q_proj = lora.Linear(d_model, d_model, r=8)
        k_proj = nn.Linear(d_model, d_model)
        v_proj = lora.Linear(d_model, d_model, r=8)
    # is then equivalent to ...
        qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        r (int, optional): Rank of the low-rank approximation. Default is 0.
        lora_alpha (int, optional): Scaling factor for the low-rank approximation. Default is 1.
        lora_dropout (float, optional): Dropout rate for LoRA. Default is 0.0.
        enable_lora (List[bool], optional): Define which of the matrices are converted. Default is [False].
        fan_in_fan_out (bool, optional): If True, the layer stores weight like (fan_in, fan_out). Default is False.
        merge_weights (bool, optional): Flag to merge weights during evaluation. Default is True.

    Raises:
        ValueError: If len(enable_lora) does not divide out_features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],  # define which of the matrices are converted.
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        if not out_features % len(enable_lora) == 0:
            raise ValueError("The length of enable_lora must divide out_features")
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        """
        Initialize A and B.
        """
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x: torch.Tensor):
        """
        Zero-pad the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Zero-padded tensor.
        """
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        """
        Merge matrices A and B.

        Returns:
            torch.Tensor: Merged tensor.
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, training_mode: bool = True):
        """
        Put the Layer into train/eval mode.

        Args:
            training_mode (bool): If True, set to training mode; else, set to evaluation mode.
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode=training_mode)
        if training_mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Compute forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


class ConvLoRA(nn.Module, LoRALayer):
    """
    Parent Class for convolutional LoRA layers.

    Args:
        conv_module (nn.Module): Convolutional module to be transformed.
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Size of the convolving kernel.
        r (int, optional): Rank of the low-rank approximation. Default is 0.
        lora_alpha (int, optional): Scaling factor for the low-rank approximation. Default is 1.
        lora_dropout (float, optional): Dropout rate for LoRA. Default is 0.0.
        merge_weights (bool, optional): Flag to merge weights during evaluation. Default is True.

    Raises:
        ValueError: If r <= 0.
    """

    def __init__(
        self,
        conv_module: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        # super calls to parent classes
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        # init lora layer
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        if not isinstance(kernel_size, int):
            logging.info(f"Kernel_size {kernel_size} was transformed into {kernel_size[0]}.")
            kernel_size = kernel_size[0]

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        else:
            raise ValueError("r should be > 0")
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        """
        Initialize A and B.
        """
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, training_mode: bool = True):
        """
        Put the Layer into train/eval mode.

        Args:
            training_mode (bool): If True, set to training mode; else, set to evaluation mode.
        """
        super(ConvLoRA, self).train(mode=training_mode)
        if training_mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Forward pass depending on if A and B have been merged.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.merged:
            return self.conv(x)
        else:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias,
            )


class LoRAConv1d(ConvLoRA):
    """
    LoRA Convolutional Layer for 1D convolutions.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super(LoRAConv1d, self).__init__(nn.Conv1d, *args, **kwargs)


class LoRAConv2d(ConvLoRA):
    """
    LoRA Convolutional Layer for 2D convolutions.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super(LoRAConv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class LoRAConv3d(ConvLoRA):
    """
    LoRA Convolutional Layer for 3D convolutions.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super(LoRAConv3d, self).__init__(nn.Conv3d, *args, **kwargs)
