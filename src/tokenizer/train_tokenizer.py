import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


TRAIN_FILE = Path("data/raw/train.jsonl")

OUTPUT_DIR = Path("tokenizer")
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"

VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "<|endoftext|>"
]

def text_iterator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text")
            if text and text.strip():
                yield text

tokenizer = Tokenizer(
    BPE(unk_token=None)
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS
)

print("Training BPE tokenizer...")

tokenizer.train_from_iterator(
    text_iterator(TRAIN_FILE),
    trainer=trainer
)

tokenizer.decoder = ByteLevelDecoder()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

tokenizer.save(
    str(OUTPUT_FILE)
)

print("Tokenizer training complete.")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"Saved to: {OUTPUT_FILE}")