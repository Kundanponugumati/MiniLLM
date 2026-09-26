import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.data.dataset import LanguageModelDataset
from src.model.gpt import GPT


# ==========================================
# CONFIG
# ==========================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 256

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
FFN_DIM = 1536

BATCH_SIZE = 8
LEARNING_RATE = 3e-4

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

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
    shuffle=True
)

x, y = next(iter(dataloader))
x = x.to(device)
y = y.to(device)

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
model = model.to(device)

# ==========================================
# LOSS + OPTIMIZER
# ==========================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# ONE TRAINING STEP
# ==========================================

optimizer.zero_grad()


# Forward pass
logits = model(x)

B, T, V = logits.shape


# Calculate loss
loss = loss_fn(
    logits.reshape(B * T, V),
    y.reshape(B * T)
)

print("Loss before update:")
print(loss.item())


# Backpropagation
loss.backward()


# Update parameters
optimizer.step()


print("\nOne training step completed.")