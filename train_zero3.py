#!/usr/bin/env python
"""
Multi-GPU training with Hydra config, Accelerate, and optional Muon optimizer.

Launch examples:
    # Vanilla baseline + DeepSpeed ZeRO-3
    accelerate launch --num_processes 2 train_zero3.py

    # Cayley HC
    accelerate launch --num_processes 2 train_zero3.py arch=cayley

    # Sinkhorn HC, 8 streams
    accelerate launch --num_processes 2 train_zero3.py arch=sinkhorn arch.hc_num_streams=8

    # Muon optimizer (DDP, no DeepSpeed)
    accelerate launch --num_processes 2 train_zero3.py use_muon=true arch=cayley

    # Override LR + enable W&B
    accelerate launch --num_processes 2 train_zero3.py lr=1e-4 wandb=true
"""

import os
import math
import shutil
import yaml
from contextlib import nullcontext

import torch
import pydantic
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from model import BaselineConfig, BaselineLM
from data import create_dataloaders, create_eval_dataloader
from train_utils import (
    save_checkpoint, load_checkpoint_before_prepare,
    resolve_resume_checkpoint, upload_accel_state_to_s3,
    init_wandb, log_wandb, finish_wandb,
    capture_data_state, restore_data_state,
    flush_async_uploads, shutdown_async_persister,
)
from hyperconnections import compute_composite_h_res_stats


# ── Pydantic config schema ───────────────────────────────────────────────

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")


class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    arch: ArchConfig

    # Data
    dataset: str = "EleutherAI/the_pile_deduplicated"
    tokenizer: str = "mistralai/Mistral-7B-v0.1"
    eval_split: str = "validation"
    eval_from_train_examples: int = 10000

    # Training
    batch_size: int = 3
    steps: int = 500000
    grad_accum: int = 43
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 3000

    # Optimizer
    use_muon: bool = False
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_matched_rms: float = 0.2

    # DeepSpeed
    ds_config: str = "ds/zero3_fp16.json"

    # Logging
    log_every: int = 50
    eval_every: int = 500
    eval_max_batches: int = 100
    eval_wikitext: bool = True
    wikitext_dataset: str = "Salesforce/wikitext"
    wikitext_config: str = "wikitext-103-raw-v1"
    wikitext_split: str = "validation"

    # Checkpoints
    save_every: int = 500
    save_dir: str = "checkpoints"
    keep_last: int = 5

    # Resume
    resume: str | None = None

    # S3 checkpoint storage
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    s3_prefix: str | None = None

    # W&B
    wandb: bool = False
    wandb_project: str = "rotary-residuals"
    wandb_entity: str | None = None
    wandb_run: str | None = None
    wandb_id: str | None = None

    # Misc
    seed: int = 42


# ── Helpers ───────────────────────────────────────────────────────────────

def _cosine_lr(step: int, warmup: int, total: int, base: float, min_ratio: float = 0.0) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def _build_model_config(config: TrainConfig) -> BaselineConfig:
    a = config.arch
    return BaselineConfig(
        d_model=a.d_model,
        n_heads=a.n_heads,
        n_layers=a.n_layers,
        d_ff=a.d_ff,
        max_seq_len=a.max_seq_len,
        gradient_checkpointing=a.gradient_checkpointing,
        tie_embeddings=False,
        hc_type=a.hc_type,
        hc_num_streams=a.hc_num_streams,
        hc_sinkhorn_tau=a.hc_sinkhorn_tau,
        hc_sinkhorn_iters=a.hc_sinkhorn_iters,
        hc_cayley_alpha=a.hc_cayley_alpha,
        hc_cayley_iters=a.hc_cayley_iters,
    )


def _build_muon_optimizer(model, config: TrainConfig):
    from muon import Muon

    muon_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]

    return Muon([
        {
            "params": muon_params,
            "use_muon": True,
            "lr": config.lr,
            "momentum": config.muon_momentum,
            "ns_steps": config.muon_ns_steps,
            "matched_adamw_rms": config.muon_matched_rms,
            "weight_decay": config.weight_decay,
        },
        {
            "params": adam_params,
            "use_muon": False,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "adamw_betas": (0.9, 0.95),
            "adamw_eps": 1e-8,
        },
    ])


def _save_resolved_config(config: TrainConfig, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    try:
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config.model_dump(), f, sort_keys=False, allow_unicode=True)
    except Exception:
        pass


def _accel_state_dir(pt_path: str) -> str:
    return pt_path.replace(".pt", "_accel")


def _get_hc_modules(model):
    """Return [(layer_idx, 'attn'|'mlp', module), ...] for all HC modules."""
    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    out = []
    for i, block in enumerate(raw.blocks):
        if getattr(block, "use_hc", False):
            out.append((i, "attn", block.hc_attn))
            out.append((i, "mlp", block.hc_mlp))
    return out


def _set_hc_collect_stats(model, enable: bool):
    for _, _, mod in _get_hc_modules(model):
        mod.collect_stats = enable


def _gather_hc_stats(model) -> dict:
    """Collect per-layer stats + composite stats into a flat wandb-friendly dict."""
    hc_mods = _get_hc_modules(model)
    stats = {}
    for layer, name, mod in hc_mods:
        for k, v in getattr(mod, "last_stats", {}).items():
            stats[f"hc/L{layer}_{name}/{k}"] = v

    all_mods = [mod for _, _, mod in hc_mods]
    composite = compute_composite_h_res_stats(all_mods)
    for k, v in composite.items():
        stats[f"hc/{k}"] = v

    return stats



# ── Main ──────────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="cfg_train", version_base=None)
def main(hydra_config: DictConfig):
    config = TrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))

    # ── Accelerator ──────────────────────────────────────────────────
    if config.use_muon:
        accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=config.grad_accum,
        )
    else:
        ds_plugin = DeepSpeedPlugin(zero_stage=3, hf_ds_config=config.ds_config)
        accelerator = Accelerator(
            mixed_precision="fp16",
            deepspeed_plugin=ds_plugin,
            gradient_accumulation_steps=config.grad_accum,
        )

    torch.manual_seed(config.seed + accelerator.process_index)

    hc_label = config.arch.hc_type if config.arch.hc_type != "none" else "vanilla"
    opt_label = "muon" if config.use_muon else "deepspeed"
    accelerator.print("=" * 60)
    accelerator.print(f"Training: Baseline Transformer ({hc_label} residuals, {opt_label})")
    accelerator.print(f"  Processes:  {accelerator.num_processes}")
    accelerator.print(f"  Precision:  {accelerator.mixed_precision}")
    accelerator.print(f"  Batch:      {config.batch_size} x {config.grad_accum} x {accelerator.num_processes}")
    if config.arch.hc_type != "none":
        accelerator.print(f"  HC streams: {config.arch.hc_num_streams}")
    accelerator.print("=" * 60)

    # ── Model ────────────────────────────────────────────────────────
    cfg = _build_model_config(config)

    ctx = accelerator.main_process_first() if accelerator.num_processes > 1 else nullcontext()
    with ctx:
        accelerator.print("Creating model...")
        model = BaselineLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        n_muon = sum(p.numel() for p in model.parameters() if p.ndim == 2)
        accelerator.print(f"Parameters: {n_params / 1e6:.2f}M  (Muon-eligible 2-D: {n_muon / 1e6:.2f}M)")

    # ── Optimizer (Muon path only; DeepSpeed manages its own) ────────
    optimizer = None
    if config.use_muon:
        optimizer = _build_muon_optimizer(model, config)
        accelerator.print(f"Muon optimizer: {len(optimizer.param_groups)} param groups")

    # ── S3 config ─────────────────────────────────────────────────────
    ckpt_prefix = f"transformer_{hc_label}_{opt_label}"
    s3_config = {
        "bucket": config.s3_bucket,
        "prefix": config.s3_prefix or ckpt_prefix,
        "access_key": config.s3_access_key,
        "secret_key": config.s3_secret_key,
        "region": config.s3_region,
        "endpoint_url": config.s3_endpoint_url,
    }

    # ── W&B ──────────────────────────────────────────────────────────
    wandb_active = init_wandb(accelerator, config, f"transformer-{hc_label}-{opt_label}", cfg, n_params)

    # ── Resume ───────────────────────────────────────────────────────
    start_step = 0
    data_state = None
    opt_state = None
    resume_path = resolve_resume_checkpoint(accelerator, config, s3_config=s3_config)
    if resume_path:
        start_step, data_state, opt_state = load_checkpoint_before_prepare(
            accelerator, model, resume_path, BaselineConfig,
        )

    # ── Data ─────────────────────────────────────────────────────────
    def run_eval(loader, max_batches):
        total_loss, total_tok = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                out = model(**batch)
                total_loss += out["loss"].item() * batch["labels"].numel()
                total_tok += batch["labels"].numel()
        if total_tok == 0:
            return None, None
        ml = total_loss / total_tok
        return ml, math.exp(min(ml, 100))

    ctx = accelerator.main_process_first() if accelerator.num_processes > 1 else nullcontext()
    with ctx:
        accelerator.print("Loading dataset...")
        train_loader, eval_loader, tokenizer = create_dataloaders(
            dataset_name=config.dataset,
            tokenizer_name=config.tokenizer,
            block_size=cfg.max_seq_len,
            batch_size=config.batch_size,
            streaming=True,
            eval_split=config.eval_split or None,
            eval_from_train_examples=config.eval_from_train_examples,
        )
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.vocab_size = len(tokenizer)

        wikitext_eval_loader = None
        if config.eval_wikitext:
            accelerator.print(f"Loading WikiText eval: {config.wikitext_dataset}")
            wikitext_eval_loader = create_eval_dataloader(
                dataset_name=config.wikitext_dataset,
                config_name=config.wikitext_config or None,
                tokenizer=tokenizer,
                block_size=cfg.max_seq_len,
                batch_size=config.batch_size,
                split=config.wikitext_split,
                streaming=True,
            )

    # ── Prepare ──────────────────────────────────────────────────────
    to_prepare = [model, train_loader]
    if optimizer is not None:
        to_prepare.append(optimizer)
    if eval_loader is not None:
        to_prepare.append(eval_loader)
    if wikitext_eval_loader is not None:
        to_prepare.append(wikitext_eval_loader)

    prepared = accelerator.prepare(*to_prepare)
    idx = 0
    model = prepared[idx]; idx += 1
    train_loader = prepared[idx]; idx += 1
    if optimizer is not None:
        optimizer = prepared[idx]; idx += 1
    if eval_loader is not None:
        eval_loader = prepared[idx]; idx += 1
    if wikitext_eval_loader is not None:
        wikitext_eval_loader = prepared[idx]; idx += 1

    restore_data_state(accelerator, train_loader, data_state)

    # Restore optimizer + scheduler state after prepare
    if resume_path and not config.use_muon:
        state_dir = _accel_state_dir(resume_path)
        if not os.path.isdir(state_dir):
            raise FileNotFoundError(
                f"Engine state directory not found: {state_dir}\n"
                "Cannot resume without optimizer + scheduler state."
            )
        accelerator.load_state(state_dir)
        accelerator.print("[resume] Restored full engine state (optimizer + scheduler)")
    elif opt_state is not None and optimizer is not None:
        optimizer.load_state_dict(opt_state)
        accelerator.print("[resume] Restored Muon optimizer state")
    del opt_state

    # Save resolved config
    if accelerator.is_main_process:
        _save_resolved_config(config, config.save_dir)

    # ── Training loop ────────────────────────────────────────────────
    accelerator.print("Starting training...")
    model.train()
    it = iter(train_loader)
    running_loss = 0.0
    tokens_local = start_step * config.batch_size * cfg.max_seq_len
    use_hc = cfg.hc_type != "none"

    for step in range(start_step + 1, config.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        # LR schedule for Muon path (DeepSpeed handles its own scheduler)
        if config.use_muon and optimizer is not None:
            lr = _cosine_lr(step, config.warmup_steps, config.steps, config.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        is_log_step = step % config.log_every == 0
        if is_log_step and use_hc:
            _set_hc_collect_stats(model, True)

        with accelerator.accumulate(model):
            out = model(**batch)
            accelerator.backward(out["loss"])

            if config.use_muon and optimizer is not None:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        hc_stats = {}
        if is_log_step and use_hc:
            hc_stats = _gather_hc_stats(model)
            _set_hc_collect_stats(model, False)

        running_loss += out["loss"].item()
        tokens_local += batch["labels"].numel()

        # ── Logging ──────────────────────────────────────────────────
        if is_log_step:
            opt_step = step // config.grad_accum
            tokens_global = int(accelerator.reduce(
                torch.tensor(tokens_local, device=accelerator.device, dtype=torch.long),
                reduction="sum",
            ).item())

        if accelerator.is_main_process and is_log_step:
            avg = running_loss / config.log_every
            current_lr = lr if config.use_muon else config.lr
            accelerator.print(
                f"step {step:6d} | opt {opt_step:6d} | tok {tokens_global:,} "
                f"| loss {avg:.4f} | lr {current_lr:.2e}"
            )
            metrics = {
                "train/loss_lm": avg,
                "train/optimizer_step": opt_step,
                "train/tokens_seen": tokens_global,
                "train/lr": current_lr,
            }
            metrics.update(hc_stats)
            log_wandb(accelerator, metrics, step, wandb_active)
            running_loss = 0.0

        # ── Evaluation ───────────────────────────────────────────────
        if (eval_loader is not None or wikitext_eval_loader is not None) and step % config.eval_every == 0:
            model.eval()
            opt_step = step // config.grad_accum
            tokens_global = int(accelerator.reduce(
                torch.tensor(tokens_local, device=accelerator.device, dtype=torch.long),
                reduction="sum",
            ).item())

            if eval_loader is not None:
                ml, ppl = run_eval(eval_loader, config.eval_max_batches)
                if ml is not None:
                    accelerator.print(f"[eval] step {step} | loss {ml:.4f} | ppl {ppl:.2f}")
                    log_wandb(accelerator, {
                        "eval/loss": ml, "eval/ppl": ppl,
                        "eval/optimizer_step": opt_step,
                        "eval/tokens_seen": tokens_global,
                    }, step, wandb_active)

            if wikitext_eval_loader is not None:
                wl, wp = run_eval(wikitext_eval_loader, config.eval_max_batches)
                if wl is not None:
                    accelerator.print(f"[eval:wikitext] step {step} | loss {wl:.4f} | ppl {wp:.2f}")
                    log_wandb(accelerator, {
                        "eval_wikitext/loss": wl, "eval_wikitext/ppl": wp,
                        "eval_wikitext/optimizer_step": opt_step,
                        "eval_wikitext/tokens_seen": tokens_global,
                    }, step, wandb_active)

            model.train()

        # ── Checkpointing ────────────────────────────────────────────
        if config.save_dir and step % config.save_every == 0:
            flush_async_uploads(accelerator)
            ds = capture_data_state(accelerator, train_loader)
            save_checkpoint(
                accelerator=accelerator, model=model, config=cfg,
                step=step, save_dir=config.save_dir,
                prefix=ckpt_prefix,
                keep_last=config.keep_last, data_state=ds,
                optimizer=optimizer, s3_config=s3_config,
            )
            if not config.use_muon:
                state_dir = os.path.join(config.save_dir, f"{ckpt_prefix}_{step}_accel")
                accelerator.save_state(state_dir)
                accelerator.print(f"[save] Engine state → {state_dir}")
                if s3_config is not None and accelerator.is_main_process:
                    upload_accel_state_to_s3(state_dir, s3_config, accelerator)

    flush_async_uploads(accelerator)
    shutdown_async_persister()
    finish_wandb(accelerator, wandb_active)
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
