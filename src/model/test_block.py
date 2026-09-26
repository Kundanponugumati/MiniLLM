import sys
from pathlib import Path

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.data.dataset import LanguageModelDataset
from src.model.embeddings import GPTEmbedding
from src.model.block import TransformerBlock


VOCAB_SIZE = 8000
CONTEXT_LENGTH = 256

D_MODEL = 384
NUM_HEADS = 6
FFN_DIM = 1536

BATCH_SIZE = 8


# ==========================================
# DATA
# ==========================================

dataset = LanguageModelDataset(
    "data/processed/train.bin",
    context_length=CONTEXT_LENGTH
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

x, y = next(iter(dataloader))


print("Token IDs:")
print(x.shape)


# ==========================================
# EMBEDDINGS
# ==========================================

embedding = GPTEmbedding(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    context_length=CONTEXT_LENGTH
)

x = embedding(x)


print("\nAfter embedding:")
print(x.shape)


# ==========================================
# TRANSFORMER BLOCK
# ==========================================

block = TransformerBlock(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM
)

x = block(x)


print("\nAfter Transformer block:")
print(x.shape)