"""
Single-GPU training with AMP (fp16).

Usage:
    python train.py
    python train.py --hc_type cayley --hc_num_streams 4
    python train.py --hc_type sinkhorn
    python train.py --hc_type fixed_rotation
    python train.py --hc_type adaptive_rotation
"""

import math
import os
import argparse
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model import BaselineConfig, BaselineLM, count_parameters
from data import create_dataloaders, create_eval_dataloader


def evaluate(model, loader, device, max_batches=200):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast("cuda", dtype=torch.float16):
                out = model(**batch)
            total_loss += out["loss"].item() * batch["labels"].numel()
            total_tok += batch["labels"].numel()
    ml = total_loss / max(total_tok, 1)
    model.train()
    return ml, math.exp(min(ml, 100))


def main():
    parser = argparse.ArgumentParser(description="Single-GPU Baseline+HC Training")

    # Data
    parser.add_argument("--dataset", default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--eval_from_train_examples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)

    # Architecture
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=1664)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # HC
    parser.add_argument("--hc_type", default="none",
                        choices=["none", "cayley", "sinkhorn", "fixed_rotation", "adaptive_rotation"])
    parser.add_argument("--hc_num_streams", type=int, default=4)
    parser.add_argument("--hc_sinkhorn_tau", type=float, default=1.0)
    parser.add_argument("--hc_sinkhorn_iters", type=int, default=20)
    parser.add_argument("--hc_cayley_alpha", type=float, default=0.1)
    parser.add_argument("--hc_cayley_iters", type=int, default=2)

    # Training
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", default="checkpoints_single")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device)

    cfg = BaselineConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        hc_type=args.hc_type, hc_num_streams=args.hc_num_streams,
        hc_sinkhorn_tau=args.hc_sinkhorn_tau, hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_cayley_alpha=args.hc_cayley_alpha, hc_cayley_iters=args.hc_cayley_iters,
    )

    print("Loading data...")
    train_loader, eval_loader, tokenizer = create_dataloaders(
        dataset_name=args.dataset, tokenizer_name=args.tokenizer,
        block_size=args.max_seq_len, batch_size=args.batch_size,
        streaming=True, eval_split=args.eval_split,
        eval_from_train_examples=args.eval_from_train_examples,
    )
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.vocab_size = len(tokenizer)

    print("Creating model...")
    model = BaselineLM(cfg).to(device)
    model.train()
    n = count_parameters(model)
    hc_label = args.hc_type if args.hc_type != "none" else "vanilla"
    print(f"Model: {n / 1e6:.2f}M params ({hc_label} residuals)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")

    it = iter(train_loader)
    acc_loss = 0.0

    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast("cuda", dtype=torch.float16):
            out = model(**batch)
            loss = out["loss"] / args.grad_accum

        scaler.scale(loss).backward()
        acc_loss += loss.item()

        if step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            opt_step = step // args.grad_accum
            if opt_step % args.log_every == 0:
                avg = acc_loss * args.grad_accum / args.log_every
                print(f"step {step} | opt {opt_step} | loss {avg:.4f}")
                acc_loss = 0.0

        if eval_loader is not None and step % args.eval_every == 0:
            ml, ppl = evaluate(model, eval_loader, device)
            print(f"[eval] step {step} | loss {ml:.4f} | ppl {ppl:.2f}")

        if args.save_dir and step % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"transformer_{hc_label}_{step}.pt")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "config": cfg,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }, path)
            print(f"[save] {path}")

    print("Done.")


if __name__ == "__main__":
    main()
