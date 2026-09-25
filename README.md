
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

