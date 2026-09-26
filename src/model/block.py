import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        dropout=0.1
    ):
        super().__init__()

        # LayerNorm before attention
        self.norm1 = nn.LayerNorm(d_model)

        # Multi-head causal self-attention
        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # LayerNorm before FFN
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout
        )


    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.norm1(x)
        )

        # FFN + residual connection
        x = x + self.ffn(
            self.norm2(x)
        )

        return x