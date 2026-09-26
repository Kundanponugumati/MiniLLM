import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.network(x)
