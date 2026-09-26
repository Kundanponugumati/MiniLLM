import torch.nn as nn

from src.model.embeddings import GPTEmbedding
from src.model.block import TransformerBlock


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_heads,
        num_layers,
        ffn_dim,
        dropout=0.1
    ):
        super().__init__()

        # ==========================================
        # EMBEDDINGS
        # ==========================================

        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            context_length=context_length,
            dropout=dropout
        )


        # ==========================================
        # TRANSFORMER BLOCKS
        # ==========================================

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])


        # ==========================================
        # FINAL LAYER NORM
        # ==========================================

        self.final_norm = nn.LayerNorm(d_model)


        # ==========================================
        # LANGUAGE MODEL HEAD
        # ==========================================

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )


    def forward(self, x):

        # (B,T)
        x = self.embedding(x)

        # (B,T,C)
        for block in self.blocks:
            x = block(x)

        # (B,T,C)
        x = self.final_norm(x)

        # (B,T,V)
        logits = self.lm_head(x)

        return logits
