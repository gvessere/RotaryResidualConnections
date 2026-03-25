#!/usr/bin/env python
"""
Text generation from a trained checkpoint.

Usage:
    python test_generate.py --checkpoint checkpoints/baseline_vanilla_1000.pt
    python test_generate.py --checkpoint checkpoints/baseline_cayley_1000.pt --prompt "Once upon a time"
"""

import argparse

import torch
from transformers import AutoTokenizer

from model import BaselineConfig, BaselineLM


def main():
    parser = argparse.ArgumentParser(description="Generate text from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    if args.greedy:
        args.temperature = 0.0

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 or args.cpu else torch.float16

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading checkpoint: {args.checkpoint}")

    torch.serialization.add_safe_globals([BaselineConfig])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", BaselineConfig())

    print(f"HC mode: {cfg.hc_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg.vocab_size = len(tokenizer)
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id

    model = BaselineLM(cfg)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {args.prompt}\n{'=' * 60}\n")

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=dtype, enabled=(device == "cuda")
    ):
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(text)
    print(f"\n{'=' * 60}")
    print(f"Generated {generated.size(1) - input_ids.size(1)} tokens")


if __name__ == "__main__":
    main()
