from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

# text = "Artificial intelligence is changing the world."
text = "Kundan built MiniLLM using PyTorch in 2026!"
encoded = tokenizer.encode(text)

print("Original:")
print(text)

print("\nTokens:")
print(encoded.tokens)

print("\nToken IDs:")
print(encoded.ids)

print("\nDecoded:")
print(tokenizer.decode(encoded.ids))
