"""Train to three epochs' worth of total updates, resuming saved progress."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.config import BATCH_SIZE, CONTEXT_LENGTH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show the target without training")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "processed" / "train.bin"
    size = data_path.stat().st_size
    if size % 2:
        raise ValueError("train.bin must contain complete uint16 tokens")
    samples = max(0, (size // 2 - 1) // CONTEXT_LENGTH)
    steps_per_epoch = samples // BATCH_SIZE
    if steps_per_epoch == 0:
        raise ValueError("Not enough training data for one complete batch")
    total_steps = 3 * steps_per_epoch
    print(f"Training samples: {samples:,}", flush=True)
    print(f"Steps per epoch: {steps_per_epoch:,}", flush=True)
    print(f"Target for 3 epochs: {total_steps:,} total steps", flush=True)
    print("Existing checkpoint progress counts toward this target.", flush=True)
    print("Resuming starts a new shuffled pass, not the exact previous batch order.", flush=True)
    if args.dry_run:
        return

    # Keep macOS awake only while this Python process is alive.
    inhibitor = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        inhibitor = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        from src.training.training import main as train
        sys.argv = [str(root / "src/training/training.py"), "--max-steps", str(total_steps)]
        train()
    finally:
        if inhibitor is not None:
            if inhibitor.poll() is None:
                inhibitor.terminate()
            inhibitor.wait()


if __name__ == "__main__":
    main()
