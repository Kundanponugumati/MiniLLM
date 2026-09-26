"""Run pretraining; automatically continue checkpoints/latest.pt when present."""
import argparse
import csv
import hashlib
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.dataset import LanguageModelDataset
from src.model.gpt import GPT
from src.training.evaluate import evaluate


def save_checkpoint(path, model, optimizer, step, best_val_loss, model_config, data_signature):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "best_val_loss": best_val_loss,
        "model_config": model_config, "data_signature": data_signature,
    }, temporary)
    temporary.replace(path)
    print(f"Saved {path.name} at step {step}", flush=True)


def load_checkpoint(path, model, optimizer, model_config, data_signature):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_config"] != model_config:
        raise ValueError("Checkpoint model settings differ from src/config.py")
    if checkpoint.get("data_signature") != data_signature:
        raise ValueError("Checkpoint tokenizer/data settings differ from this run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"], checkpoint["best_val_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS,
                        help="Total completed-step target, including resumed steps")
    parser.add_argument("--eval-interval", type=int, default=config.EVAL_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=config.SAVE_INTERVAL)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.max_steps, args.eval_interval, args.save_interval, args.eval_batches) <= 0:
        parser.error("Step counts and intervals must be positive")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}", flush=True)
    torch.manual_seed(42)

    train_path = PROJECT_ROOT / "data/processed/train.bin"
    val_path = PROJECT_ROOT / "data/processed/validation.bin"
    tokenizer_path = PROJECT_ROOT / "tokenizer/tokenizer.json"
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.VOCAB_SIZE:
        raise ValueError("Tokenizer vocabulary does not match model vocabulary")
    train_data = LanguageModelDataset(train_path, config.CONTEXT_LENGTH)
    val_data = LanguageModelDataset(val_path, config.CONTEXT_LENGTH)
    if len(train_data) < config.BATCH_SIZE or len(val_data) == 0:
        raise ValueError("Not enough training or validation tokens")
    for dataset in (train_data, val_data):
        if int(dataset.data.max()) >= config.VOCAB_SIZE:
            raise ValueError("Dataset contains token IDs outside the vocabulary")
    signature = {
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "train_size": train_path.stat().st_size,
        "train_modified": train_path.stat().st_mtime_ns,
        "validation_size": val_path.stat().st_size,
        "validation_modified": val_path.stat().st_mtime_ns,
        "batch_size": config.BATCH_SIZE,
        "eval_batches": args.eval_batches,
    }
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=True)
    # A repeatable sample spread across the validation split.
    count = min(len(val_data), args.eval_batches * config.BATCH_SIZE)
    indices = torch.randperm(len(val_data), generator=torch.Generator().manual_seed(123))[:count].tolist()
    val_loader = DataLoader(Subset(val_data, indices), batch_size=config.BATCH_SIZE)
    model = GPT(**config.MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir / "latest.pt"
    step, best = 0, float("inf")
    if latest.exists():
        step, best = load_checkpoint(latest, model, optimizer, config.MODEL_CONFIG, signature)
        print(f"Resumed after step {step} (new shuffled data pass)", flush=True)
    elif (args.checkpoint_dir / "best.pt").exists():
        raise ValueError("best.pt exists without latest.pt; choose a new checkpoint directory")
    print(f"Training samples: {len(train_data):,} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Target: {args.max_steps} steps | Validation: {count} sequences | Checkpoints: {args.checkpoint_dir}", flush=True)
    if step >= args.max_steps:
        print("Target already reached. Use --max-steps with a higher value to continue.")
        return

    def save(name):
        save_checkpoint(args.checkpoint_dir / name, model, optimizer, step, best,
                        config.MODEL_CONFIG, signature)

    log_path = args.checkpoint_dir / "metrics.csv"
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    started = time.monotonic()
    starting_step = step
    model.train()
    with log_path.open("a", newline="") as log:
        writer = csv.writer(log)
        if needs_header:
            writer.writerow(["step", "training_loss", "validation_loss", "elapsed_seconds"])
        try:
            while step < args.max_steps:
                for x, y in train_loader:
                    if step >= args.max_steps:
                        break
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, config.VOCAB_SIZE), y.reshape(-1))
                    train_loss = loss.item()
                    if not math.isfinite(train_loss):
                        raise RuntimeError(f"Non-finite training loss before step {step + 1}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    val_loss = ""
                    if step % args.eval_interval == 0 or step == args.max_steps:
                        val_loss = evaluate(model, val_loader, device, max_batches=None)
                        if not math.isfinite(val_loss):
                            raise RuntimeError("Non-finite validation loss")
                        print(f"Validation at step {step}: {val_loss:.4f}", flush=True)
                        if val_loss < best:
                            best = val_loss
                            save("best.pt")
                    if step % args.save_interval == 0 or step == args.max_steps:
                        save("latest.pt")
                    elapsed = time.monotonic() - started
                    writer.writerow([step, train_loss, val_loss, round(elapsed, 2)])
                    log.flush()
                    if step == starting_step + 1 or step % 10 == 0:
                        rate = (step - starting_step) / elapsed
                        eta = (args.max_steps - step) / rate / 60
                        print(f"Step {step:05d} | Loss: {train_loss:.4f} | {rate:.2f} steps/s | ETA: {eta:.1f} min", flush=True)
        except KeyboardInterrupt:
            # Keep the last fully saved update; an interrupt can occur inside AdamW.
            print("\nStopped. Restart the same command to resume the last saved checkpoint.", flush=True)
            return
    print(f"Training complete. Best sampled validation loss: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
