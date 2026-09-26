import torch
import torch.nn as nn
from math import sqrt


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads


        self.W_query = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_key = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_value = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.out_projection = nn.Linear(
            d_model,
            d_model,
            bias=False
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape


        # Q, K, V projections
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)


        # Split into heads
        Q = Q.view(
            B, T,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            B, T,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            B, T,
            self.num_heads,
            self.head_dim
        )


        # (B,T,H,D) → (B,H,T,D)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / sqrt(self.head_dim)


        # Causal mask
        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )


        # Attention probabilities
        attention_weights = torch.softmax(scores,dim=-1)
        attention_weights = self.attention_dropout(attention_weights)


        # Weighted values
        context = attention_weights @ V


        # (B,H,T,D) → (B,T,H,D)
        context = context.transpose(1, 2)


        # Merge heads
        context = context.contiguous().view(
            B,
            T,
            C
        )


        # Mix information from heads
        output = self.out_projection(context)
        output = self.output_dropout(output)

        return output
