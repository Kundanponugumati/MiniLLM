import sys
from pathlib import Path

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.data.dataset import LanguageModelDataset
from src.model.embeddings import GPTEmbedding
from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward


VOCAB_SIZE = 8000
CONTEXT_LENGTH = 256

D_MODEL = 384
NUM_HEADS = 6
FFN_DIM = 1536

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


x, y = next(iter(dataloader))

print("Token IDs:")
print(x.shape)


# Embeddings
embedding = GPTEmbedding(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    context_length=CONTEXT_LENGTH
)

x = embedding(x)

print("\nAfter embedding:")
print(x.shape)


# Attention
attention = CausalSelfAttention(
    d_model=D_MODEL,
    num_heads=NUM_HEADS
)

x = attention(x)

print("\nAfter attention:")
print(x.shape)


# Feed Forward
ffn = FeedForward(
    d_model=D_MODEL,
    ffn_dim=FFN_DIM
)

x = ffn(x)

print("\nAfter feed forward:")
print(x.shape)