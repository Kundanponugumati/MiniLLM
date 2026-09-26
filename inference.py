"""Generate a text continuation using a saved MiniLLM checkpoint."""
import argparse
import hashlib
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.model.gpt import GPT


@torch.inference_mode()
def generate(model, token_ids, context_length, max_new_tokens, temperature, top_k, eos_id):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(token_ids[:, -context_length:])[:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                selection = torch.multinomial(torch.softmax(values, dim=-1), 1)
                next_token = indices.gather(-1, selection)
            else:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if next_token.item() == eos_id:
            break
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids


def chat(model, tokenizer, model_config, args, device):
    """Keep the model loaded across turns; history stays in memory only."""
    history = ""
    context_length = model_config["context_length"]
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    print("\nMiniLLM chat — /clear to reset, /exit to quit.")
    print("This pretrained model may continue text instead of answering questions.")
    print(f"Only the latest {context_length} tokens fit in its context.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if message.lower() in ("/exit", "/quit"):
            print("Goodbye!")
            return
        if message.lower() == "/clear":
            history = ""
            print("Conversation cleared.\n")
            continue
        if not message:
            continue
        prompt = f"{history}User: {message}\nAssistant:"
        ids = tokenizer.encode(prompt).ids[-context_length:]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        print("MiniLLM: ", end="", flush=True)
        try:
            result = generate(model, tokens, context_length, args.max_new_tokens,
                              args.temperature, args.top_k, eos_id)
        except KeyboardInterrupt:
            print("[Generation cancelled]\n")
            continue
        reply = tokenizer.decode(result[0, len(ids):].tolist())
        # Do not display a model-invented next user turn as the user's message.
        for marker in ("\nUser:", "\nAssistant:"):
            reply = reply.split(marker, 1)[0]
        reply = reply.strip()
        print((reply or "[No reply generated; try another prompt.]") + "\n")
        if reply:
            history_ids = tokenizer.encode(prompt + " " + reply + "\n").ids
            history = tokenizer.decode(history_ids[-context_length:])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The solar system consists of")
    parser.add_argument("--chat", action="store_true", help="Start an interactive terminal chat")
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints/best.pt")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Use 0 for deterministic greedy decoding")
    parser.add_argument("--top-k", type=int, default=40, help="Use 0 to sample from all tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.top_k < 0:
        parser.error("max-new-tokens must be positive and top-k must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    tokenizer_path = root / "tokenizer/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    expected_hash = checkpoint.get("data_signature", {}).get("tokenizer_sha256")
    if expected_hash and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != expected_hash:
        parser.error("Tokenizer does not match the checkpoint")
    if tokenizer.get_vocab_size() != model_config["vocab_size"]:
        parser.error("Tokenizer vocabulary does not match the checkpoint")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    model = GPT(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    step = checkpoint["step"]
    del checkpoint
    torch.manual_seed(args.seed)
    print(f"Loaded {args.checkpoint.name} | step {step:,} | device: {device}", flush=True)
    if args.chat:
        chat(model, tokenizer, model_config, args, device)
        return
    ids = tokenizer.encode(args.prompt).ids
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if not ids:
        if eos_id is None:
            parser.error("Supply a nonempty prompt")
        ids = [eos_id]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    result = generate(model, tokens, model_config["context_length"],
                      args.max_new_tokens, args.temperature, args.top_k, eos_id)
    print("\n" + tokenizer.decode(result[0].tolist()), flush=True)


if __name__ == "__main__":
    main()
