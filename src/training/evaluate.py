"""Token-weighted validation loss; preserves the caller's training mode."""
from itertools import islice

import torch
import torch.nn.functional as F


def evaluate(model, dataloader, device, max_batches=20):
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        batches = dataloader if max_batches is None else islice(dataloader, max_batches)
        with torch.no_grad():
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
                )
                total_loss += loss.item()
                total_tokens += y.numel()
        if not total_tokens:
            raise ValueError("Validation loader contains no tokens")
        return total_loss / total_tokens
    finally:
        model.train(was_training)
