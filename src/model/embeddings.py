import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            context_length,
            d_model
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T = x.shape
        token_embeddings = self.token_embedding(x)
        positions = torch.arange(
            T,
            device=x.device
        )
        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings
        return self.dropout(x)
