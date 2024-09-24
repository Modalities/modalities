import torch
from torch import nn

from modalities.models.coca.multi_modal_decoder import TransformerBlock
from modalities.models.gpt2.gpt2_model import ActivationType
from modalities.models.model import NNModel
from modalities.nn.attention import AttentionConfig, AttentionType


class TextDecoder(NNModel):
    """TextDecoder class."""

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation: ActivationType,
        epsilon: float,
        attention_config: AttentionConfig = None,
    ):
        """
        Initializes the TextDecoder class.

        Args:
            sample_key (str): The key for the samples.
            prediction_key (str): The key for the predictions.
            block_size (int): The block size.
            vocab_size (int): The size of the vocabulary.
            n_layer (int): The number of layers.
            n_head (int): The number of attention heads.
            n_embd (int): The embedding dimension.
            ffn_hidden (int): The hidden dimension of the feed-forward network.
            dropout (float): The dropout rate.
            bias (bool): Flag indicating whether to include bias terms.
            activation (ActivationType): The activation function to use.
            epsilon (float): Small value to avoid division by zero in LayerNorm.
            attention_config (AttentionConfig, optional): The attention configuration. Defaults to None.
        """
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size

        self.cls_token = nn.Parameter(torch.empty(1, 1, n_embd))
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                wpe=nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        TransformerBlock(
                            n_embd=n_embd,
                            bias=bias,
                            epsilon=epsilon,
                            activation=activation,
                            n_head=n_head,
                            dropout=dropout,
                            ffn_hidden=ffn_hidden,
                            with_context=False,
                            attention_type=AttentionType.CAUSAL_SELF_ATTENTION,
                            attention_config=attention_config,
                        )
                        for _ in range(n_layer)
                    ]
                ),
            )
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TextDecoder module.

        Args:
            inputs (dict[str, torch.Tensor]): Input dictionary.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the predictions.
        """
        input_ids = inputs[self.sample_key]
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T + 1, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(input_ids)
        tok_emb = torch.cat([tok_emb, self.cls_token.repeat(B, 1, 1)], dim=1)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        return {self.prediction_key: x}
