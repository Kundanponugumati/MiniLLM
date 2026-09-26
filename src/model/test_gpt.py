import sys
from pathlib import Path

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.data.dataset import LanguageModelDataset
from src.model.gpt import GPT


VOCAB_SIZE = 8000
CONTEXT_LENGTH = 256

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
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


print("Input:")
print(x.shape)


# ==========================================
# MODEL
# ==========================================

model = GPT(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ffn_dim=FFN_DIM
)


logits = model(x)


print("\nInput token IDs:")
print(x.shape)

print("\nLogits:")
print(logits.shape)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTotal parameters:")
print(f"{total_params:,}")

