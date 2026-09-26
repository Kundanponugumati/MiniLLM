"""Shared defaults for pretraining."""
VOCAB_SIZE = 8000
CONTEXT_LENGTH = 256
D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MAX_STEPS = 5000
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
MAX_GRAD_NORM = 1.0

MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ffn_dim=D_FF,
    dropout=DROPOUT,
)
