
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Dataset / DataLoader
 ↓
Embeddings
 ↓
Positional Embeddings
 ↓
Causal Multi-Head Attention
 ↓
Feed Forward
 ↓
Transformer Blocks × N
 ↓
LM Head
 ↓
Logits
 ↓
Next-token prediction


vocab_size
    = How many DIFFERENT tokens exist overall.

B / batch_size
    = How many sequences we're processing together.

T / context_length
    = How many token positions are in each sequence.

C / embedding_dim / d_model
    = How many numbers represent each token.


PART 1 — PROJECT SETUP
        ↓
PART 2 — REAL DATASET
        ↓
PART 3 — TOKENIZER
        ↓
PART 4 — TRAIN / VALIDATION SPLIT
        ↓
PART 5 — DATASET
        ↓
PART 6 — DATALOADER
        ↓
PART 7 — TOKEN EMBEDDING
        ↓
PART 8 — POSITION INFORMATION
        ↓
PART 9 — CAUSAL SELF-ATTENTION
        ↓
PART 10 — MULTI-HEAD ATTENTION
        ↓
PART 11 — FEED-FORWARD NETWORK
        ↓
PART 12 — LAYER NORM + RESIDUALS
        ↓
PART 13 — TRANSFORMER BLOCK
        ↓
PART 14 — STACK TRANSFORMER BLOCKS
        ↓
PART 15 — LM HEAD
        ↓
PART 16 — COMPLETE GPT MODEL
        ↓
        ↓
========== TRAINING ==========
        ↓
PART 17 — LOSS
        ↓
PART 18 — OPTIMIZER
        ↓
PART 19 — TRAINING LOOP
        ↓
PART 20 — VALIDATION
        ↓
PART 21 — CHECKPOINTING
        ↓
PART 22 — TRAIN THE MODEL
        ↓
        ↓
========== INFERENCE =========
        ↓
PART 23 — LOAD CHECKPOINT
        ↓
PART 24 — GENERATION LOOP
        ↓
PART 25 — TEMPERATURE / TOP-K
        ↓
PART 26 — GENERATE TEXT



MODEL
≈ 20–30M parameters

ARCHITECTURE
Decoder-only Transformer

DATASET
FineWeb-Edu

RAW LOCAL CORPUS
≈ 500 MB maximum initially

TOKENIZER
BPE

VOCAB
≈ 8,000

CONTEXT
256

D_MODEL
384

HEADS
6

LAYERS
6

FFN
1536

DEVICE
Apple MPS where appropriate

STORAGE BUDGET
Keep entire project comfortably <10 GB

TRAINING STRATEGY
tiny sanity run → small run → full planned run
## Pretraining

From the project root, run:

```bash
caffeinate -i python -u src/training/training.py --max-steps 5000
```

On macOS, `caffeinate` prevents idle sleep while training runs. Keep the terminal open.
The script selects CUDA, Apple MPS, or CPU automatically. Defaults live in
`src/config.py`: 8,000 vocabulary entries, context 256, width 384, six heads,
six blocks, FFN 1536, dropout 0.1, batch size 8, AdamW learning rate 3e-4.

Training validates every 250 steps on a fixed random sample of 160 validation
sequences, saves `checkpoints/latest.pt` every 500 steps and at completion,
and saves `checkpoints/best.pt` whenever sampled validation loss improves.
`checkpoints/metrics.csv` records training and validation losses and elapsed
seconds within each invocation. Progress output includes throughput and ETA.

Running the same command automatically restores model weights, AdamW state,
and completed step count from `latest.pt`. `--max-steps` is the total target,
not additional steps. To continue beyond 5,000, use `--max-steps 10000`.
Resume uses a new shuffled data pass; it is not an exact replay of the original
batch order or random state. Model settings, tokenizer fingerprint, and data
file metadata are checked before resuming. Keep the tokenizer and processed
data unchanged. To start a separate run, use `--checkpoint-dir checkpoints/new-run`.

Ctrl+C stops training; restart to recover the last saved checkpoint. Updates
since that checkpoint may be lost. Checkpoints are written through a temporary
file before replacing the previous saved file. For a short recovery test, use
a separate checkpoint directory with `--max-steps 2 --eval-interval 1
--save-interval 1`, then repeat with `--max-steps 3`.
