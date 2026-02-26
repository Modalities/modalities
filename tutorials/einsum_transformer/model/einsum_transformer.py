from typing import overload

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, mlp_expansion_factor: float = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.num_q_heads_per_kv_head = num_q_heads // num_kv_heads

        self.W_q = nn.Parameter(torch.zeros(size=(embed_dim, self.num_q_heads, self.head_dim)))
        self.W_k = nn.Parameter(torch.zeros(size=(embed_dim, num_kv_heads, self.head_dim)))
        self.W_v = nn.Parameter(torch.zeros(size=(embed_dim, num_kv_heads, self.head_dim)))
        self.W_o = nn.Parameter(torch.zeros(size=(self.num_q_heads, self.head_dim, embed_dim)))  # NHD

        self.mlp = MLP(embed_dim, mlp_expansion_factor)
        self.pre_attn_norm = RMSNorm(embed_dim)
        self.pre_mlp_norm = RMSNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x_new = self.pre_attn_norm(x)  # BTD
        q = torch.einsum("BTD,DNH->BTNH", x_new, self.W_q)  # BTNH
        q = q.reshape(
            shape=(batch_size, seq_len, self.num_kv_heads, self.num_q_heads_per_kv_head, self.head_dim)
        )  # BTKGH

        k = torch.einsum("BTD,DKH->BTKH", x_new, self.W_k)
        a = torch.einsum("BTKGH,BSKH->BTSKG", q, k)  # BTTKG (here, T=S)

        # apply causal mask
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=x.device, dtype=a.dtype
        )  # T by T matrix with -inf entries
        mask = torch.triu(mask, diagonal=1)[
            None, :, :, None, None
        ]  # only keep upper triangular part (future positions), 1TT11
        a = a + mask  # BTTKG

        # attention scores -> probabilities
        a = a / (self.head_dim**0.5)
        a = torch.softmax(a, dim=2)  # softmax over S

        v = torch.einsum("BTD,DKH->BTKH", x_new, self.W_v)
        x_new = torch.einsum("BTSKG,BSKH->BTKGH", a, v)  # BTKGH
        x_new = x_new.reshape(
            shape=(batch_size, seq_len, self.num_kv_heads * self.num_q_heads_per_kv_head, self.head_dim)
        )  # BTNH
        x_new = torch.einsum("BTNH,NHD->BTD", x_new, self.W_o)  # BTD

        x_new = x_new + x  # residual connection

        x_new = self.pre_mlp_norm(x_new)
        x_new = x_new + self.mlp(x_new)  # residual connection
        return x_new

    def reset_parameters(self):
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_k, std=0.02)
        nn.init.normal_(self.W_v, std=0.02)
        nn.init.normal_(self.W_o, std=0.02)
        self.mlp.reset_parameters()


class MLP(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: float = 4):
        super().__init__()
        upscaled_dim = int(embed_dim * expansion_factor)
        self.W_in1 = nn.Parameter(torch.zeros(size=(embed_dim, upscaled_dim)))
        self.W_in2 = nn.Parameter(torch.zeros(size=(embed_dim, upscaled_dim)))
        self.W_out = nn.Parameter(torch.zeros(size=(upscaled_dim, embed_dim)))
        self.gelu = nn.GELU()

    def forward(self, x):
        x_new1 = torch.einsum("BTD,DF->BTF", x, self.W_in1)  # BTF
        x_new1 = self.gelu(x_new1)
        x_new2 = torch.einsum("BTD,DF->BTF", x, self.W_in2)  # BTF
        x_new = x_new1 + x_new2  # BTF
        x_new = torch.einsum("BTF,FD->BTD", x_new, self.W_out)  # BTD
        return x_new

    def reset_parameters(self):
        nn.init.normal_(self.W_in1, std=0.02)
        nn.init.normal_(self.W_in2, std=0.02)
        nn.init.normal_(self.W_out, std=0.02)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps=None, device=None, dtype=None):  # reusing the pytorch function signature
        super().__init__()
        self.eps = eps if eps is not None else 1e-8  # for numerical stability
        self.gain = nn.Parameter(torch.ones(size=(embed_dim,), dtype=dtype, device=device))

    def forward(self, x):  # input shape: BTD
        x = x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / x.shape[-1] + self.eps)  # RMS normalization
        x = x * self.gain  # elementwise scaling
        return x  # BTD

    def reset_parameters(self):
        nn.init.ones_(self.gain)


class EinsumTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_layers: int,
        mlp_expansion_factor: int = 4,
        sample_key: str = "input_ids",
        prediction_key: str = "logits",
    ):
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key

        self.vocab_size = vocab_size
        self.Wte = nn.Parameter(torch.zeros(size=(vocab_size, embed_dim)))
        self.Wpe = nn.Parameter(torch.zeros(size=(sequence_length, embed_dim)))
        self.pre_lm_head_norm = RMSNorm(embed_dim)

        self.lm_head = nn.Parameter(torch.zeros(size=(embed_dim, vocab_size)))

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_q_heads, num_kv_heads, mlp_expansion_factor) for _ in range(num_layers)]
        )

    @property
    def weight_decay_groups(self):
        return {
            "linear": [".W_q", ".W_k", ".W_v", ".W_o", ".W_in1", ".W_in2", ".W_out", "lm_head"],
            "embedding": ["Wte", "Wpe"],
            "layernorm": [".pre_lm_head_norm", ".pre_attn_norm", ".pre_mlp_norm"],
        }

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the GPT2LLM module.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.
                - sample_key (str): Key for the input tensor containing token ids.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing output tensors.
                - prediction_key (str): Key for the output tensor containing logits.
        """
        ...

    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            inputs (torch.Tensor): A tensor containing input token ids.

        Returns:
            torch.Tensor: A tensor containing output logits.
        """
        ...

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass of the module.

        Args:
            inputs (dict[str, torch.Tensor] | torch.Tensor): Input data.

        Returns:
            dict[str, torch.Tensor] | torch.Tensor: Model output.
        """
        if isinstance(inputs, dict):
            return {self.prediction_key: self.forward_impl(inputs[self.sample_key])}
        else:
            return self.forward_impl(inputs)

    def forward_impl(self, x: torch.Tensor):
        word_embeddings = self.Wte[x]  # BTD
        pos_indices = torch.arange(start=0, end=x.shape[1], device=x.device)[None, :]  # 1T
        pos_embeddings = self.Wpe[pos_indices, :]  # 1TD
        x = word_embeddings + pos_embeddings  # BTD
        # transformer layers
        for layer in self.layers:
            x = layer(x)
        # lm head
        x = self.pre_lm_head_norm(x)  # BTD
        x = torch.einsum("BTD,DV->BTV", x, self.lm_head)  # BTV
        return x

    def reset_parameters(self):
        nn.init.normal_(self.Wte, std=0.02)
        nn.init.normal_(self.Wpe, std=0.02)
        nn.init.normal_(self.lm_head, std=0.02)
        self.pre_lm_head_norm.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


if __name__ == "__main__":
    BATCH_SIZE = 2  # B
    SEQ_LEN = 64  # T
    EMBED_DIM = 512  # D
    NUM_Q_HEADS = 8  # N
    NUM_KV_HEADS = 2  # K
    VOCAB_SIZE = 100  # V

    NUM_LAYERS = 6

    transformer = EinsumTransformer(
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_layers=NUM_LAYERS,
    )
    transformer.reset_parameters()
    x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    transformer(x)
    print(x)
