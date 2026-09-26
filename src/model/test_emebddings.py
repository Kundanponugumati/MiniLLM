import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.data.dataset import LanguageModelDataset
from src.model.embeddings import GPTEmbedding


VOCAB_SIZE = 8000
CONTEXT_LENGTH = 256
D_MODEL = 384
BATCH_SIZE = 8


# Dataset
dataset = LanguageModelDataset(
    "data/processed/train.bin",
    context_length=CONTEXT_LENGTH
)


# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Get one batch
x, y = next(iter(dataloader))

embedding = GPTEmbedding(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    context_length=CONTEXT_LENGTH
)

x_emb = embedding(x)

print("Input:")
print(x.shape)

print("\nAfter token + position embedding:")
print(x_emb.shape)
