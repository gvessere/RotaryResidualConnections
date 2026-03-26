"""
Shared training utilities: checkpointing, S3 storage, W&B logging, CLI args, resume.
"""

import os
import re
import glob
import shutil
import tarfile
from typing import Optional, Any, Dict, Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── S3 helpers ────────────────────────────────────────────────────────────

def _build_s3_client(s3_config: Dict[str, Any]):
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        region_name=s3_config.get("region"),
        endpoint_url=s3_config.get("endpoint_url"),
    )


def _upload_to_s3(local_path: str, s3_config: Dict[str, Any], accelerator) -> str:
    client = _build_s3_client(s3_config)
    bucket = s3_config["bucket"]
    prefix = s3_config.get("prefix", "checkpoints")
    key = f"{prefix}/{os.path.basename(local_path)}"
    client.upload_file(local_path, bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    accelerator.print(f"[s3] Uploaded → {s3_uri}")
    return s3_uri


def _download_from_s3(s3_uri: str, local_dir: str, s3_config: Dict[str, Any], accelerator) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    client = _build_s3_client(s3_config)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(key))
    accelerator.print(f"[s3] Downloading {s3_uri} → {local_path}")
    client.download_file(bucket, key, local_path)
    return local_path


def upload_accel_state_to_s3(state_dir: str, s3_config: Dict[str, Any], accelerator):
    """Tar+gzip an accelerator state directory and upload it to S3."""
    tar_path = state_dir + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(state_dir, arcname=os.path.basename(state_dir))
    try:
        _upload_to_s3(tar_path, s3_config, accelerator)
    finally:
        os.remove(tar_path)


def _download_and_extract_accel_tar(
    s3_uri: str, local_dir: str, s3_config: Dict[str, Any], accelerator,
) -> str:
    """Download a compressed accel state tarball from S3 and extract it."""
    tar_path = _download_from_s3(s3_uri, local_dir, s3_config, accelerator)
    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            tar.extractall(path=local_dir, filter="data")
        except TypeError:
            tar.extractall(path=local_dir)
    extracted = tar_path.removesuffix(".tar.gz")
    os.remove(tar_path)
    accelerator.print(f"[s3] Extracted accel state → {extracted}")
    return extracted


def _cleanup_old_s3_checkpoints(s3_config, ckpt_prefix, keep_last, accelerator):
    client = _build_s3_client(s3_config)
    bucket = s3_config["bucket"]
    s3_prefix = s3_config.get("prefix", "checkpoints")
    full_prefix = f"{s3_prefix}/{ckpt_prefix}_"

    paginator = client.get_paginator("list_objects_v2")
    pt_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pt"):
                pt_objects.append(obj)

    if len(pt_objects) <= keep_last:
        return

    def _step(obj):
        m = re.search(r"_(\d+)\.pt$", obj["Key"])
        return int(m.group(1)) if m else 0

    for old in sorted(pt_objects, key=_step, reverse=True)[keep_last:]:
        try:
            client.delete_object(Bucket=bucket, Key=old["Key"])
            tar_key = old["Key"].replace(".pt", "_accel.tar.gz")
            client.delete_object(Bucket=bucket, Key=tar_key)
            accelerator.print(f"[s3:cleanup] Deleted s3://{bucket}/{old['Key']}")
        except Exception:
            pass


# ── Checkpointing ────────────────────────────────────────────────────────

def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: Any,
    step: int,
    save_dir: str,
    prefix: str = "checkpoint",
    keep_last: int = 5,
    data_state: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    s3_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    accelerator.wait_for_everyone()
    os.makedirs(save_dir, exist_ok=True)
    state_dict = accelerator.get_state_dict(model)

    opt_state = None
    if optimizer is not None:
        opt_state = optimizer.state_dict()

    data_state_by_rank = None
    if data_state is not None and dist.is_available() and dist.is_initialized():
        try:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, data_state)
            data_state_by_rank = gathered
        except Exception as e:
            accelerator.print(f"[resume] Could not gather per-rank data state: {e}")

    ckpt_path = None
    if accelerator.is_main_process:
        ckpt_path = os.path.join(save_dir, f"{prefix}_{step}.pt")
        payload: Dict[str, Any] = {"step": step, "model": state_dict, "config": config}
        if opt_state is not None:
            payload["optimizer"] = opt_state
        if data_state is not None:
            payload["data_state"] = data_state
        if data_state_by_rank is not None:
            payload["data_state_by_rank"] = data_state_by_rank
        try:
            import wandb as _wb
            if _wb.run is not None:
                payload["wandb_run_id"] = _wb.run.id
                payload["wandb_run_name"] = _wb.run.name
        except Exception:
            pass
        torch.save(payload, ckpt_path)
        accelerator.print(f"[save] Checkpoint saved to {ckpt_path}")

        if keep_last > 0:
            _cleanup_old_checkpoints(save_dir, prefix, keep_last, accelerator)

        if s3_config is not None:
            try:
                _upload_to_s3(ckpt_path, s3_config, accelerator)
                if keep_last > 0:
                    _cleanup_old_s3_checkpoints(s3_config, prefix, keep_last, accelerator)
            except Exception as e:
                accelerator.print(f"[s3] Upload failed: {e}")

    return ckpt_path


def _cleanup_old_checkpoints(save_dir, prefix, keep_last, accelerator):
    pattern = os.path.join(save_dir, f"{prefix}_*.pt")
    ckpts = glob.glob(pattern)
    if len(ckpts) <= keep_last:
        return

    def _step(p):
        m = re.search(rf"{prefix}_(\d+)\.pt$", p)
        return int(m.group(1)) if m else 0

    for old in sorted(ckpts, key=_step, reverse=True)[keep_last:]:
        try:
            os.remove(old)
            state_dir = old.replace(".pt", "_accel")
            if os.path.isdir(state_dir):
                shutil.rmtree(state_dir)
            accelerator.print(f"[cleanup] Removed {os.path.basename(old)}")
        except OSError:
            pass


# ── Loading / resume ─────────────────────────────────────────────────────

def load_checkpoint_before_prepare(
    accelerator: Accelerator,
    model: torch.nn.Module,
    checkpoint_path: str,
    config_class: type,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    accelerator.print(f"Loading checkpoint: {checkpoint_path}")
    torch.serialization.add_safe_globals([config_class])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    empty = sum(1 for v in state_dict.values() if v.numel() == 0)
    if empty > 10:
        accelerator.print(f"WARNING: {empty} empty tensors (bad ZeRO-3 save)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        accelerator.print(f"Missing keys: {len(missing)}")
    if unexpected:
        accelerator.print(f"Unexpected keys: {len(unexpected)}")

    step = ckpt.get("step", 0)
    data_state = _extract_local_data_state(accelerator, ckpt)
    opt_state = ckpt.get("optimizer")
    accelerator.print(f"Resumed from step {step}"
                      f"{' (with optimizer state)' if opt_state else ' (no optimizer state)'}")
    return step, data_state, opt_state


def capture_data_state(accelerator, train_loader):
    try:
        if hasattr(train_loader, "state_dict"):
            s = train_loader.state_dict()
            if isinstance(s, dict) and s:
                return s
    except Exception:
        pass
    try:
        ds = getattr(train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "state_dict"):
            s = ds.state_dict()
            if isinstance(s, dict) and s:
                return s
    except Exception:
        pass
    return None


def _extract_local_data_state(accelerator, ckpt):
    by_rank = ckpt.get("data_state_by_rank")
    if isinstance(by_rank, list):
        rank = accelerator.process_index
        if rank < len(by_rank) and isinstance(by_rank[rank], dict):
            return by_rank[rank]
    shared = ckpt.get("data_state")
    if accelerator.num_processes > 1 and shared is not None:
        return None
    return shared if isinstance(shared, dict) else None


def restore_data_state(accelerator, train_loader, data_state):
    if not data_state:
        return False
    try:
        if hasattr(train_loader, "load_state_dict"):
            train_loader.load_state_dict(data_state)
            accelerator.print("[resume] Restored dataloader state")
            return True
    except Exception:
        pass
    try:
        ds = getattr(train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "load_state_dict"):
            ds.load_state_dict(data_state)
            accelerator.print("[resume] Restored dataset state")
            return True
    except Exception:
        pass
    return False


def resolve_resume_checkpoint(accelerator, args, s3_config=None):
    resume = getattr(args, "resume", None)
    if not resume:
        return None
    if not resume.startswith("s3://"):
        return resume

    if s3_config is None:
        raise RuntimeError("S3 credentials required to resume from an s3:// path")

    save_root = getattr(args, "save_dir", None) or "."
    os.makedirs(save_root, exist_ok=True)
    shared_path = os.path.join(save_root, ".resume_s3_path")

    if accelerator.is_main_process:
        local = _download_from_s3(resume, save_root, s3_config, accelerator)
        tar_uri = resume.replace(".pt", "_accel.tar.gz")
        _download_and_extract_accel_tar(tar_uri, save_root, s3_config, accelerator)
        with open(shared_path, "w") as f:
            f.write(local)

    accelerator.wait_for_everyone()
    with open(shared_path) as f:
        return f.read().strip()


# ── CLI args ──────────────────────────────────────────────────────────────

def get_common_args(parser, default_save_dir="checkpoints"):
    parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_from_train_examples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=3)

    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--grad_accum", type=int, default=43)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_max_batches", type=int, default=100)
    parser.add_argument("--eval_wikitext", dest="eval_wikitext", action="store_true", default=True)
    parser.add_argument("--no_eval_wikitext", dest="eval_wikitext", action="store_false")
    parser.add_argument("--wikitext_dataset", type=str, default="Salesforce/wikitext")
    parser.add_argument("--wikitext_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--wikitext_split", type=str, default="validation")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument("--keep_last", type=int, default=5)

    parser.add_argument("--ds_config", type=str, default="ds/zero3_fp16.json")

    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--s3_bucket", type=str, required=True)
    parser.add_argument("--s3_access_key", type=str, required=True)
    parser.add_argument("--s3_secret_key", type=str, required=True)
    parser.add_argument("--s3_region", type=str, default=None)
    parser.add_argument("--s3_endpoint_url", type=str, default=None)
    parser.add_argument("--s3_prefix", type=str, default=None)

    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="hyper-baseline")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    return parser


# ── W&B helpers ───────────────────────────────────────────────────────────

def init_wandb(accelerator, args, model_name, config, n_params):
    if not args.wandb or not WANDB_AVAILABLE:
        if args.wandb and not WANDB_AVAILABLE:
            accelerator.print("WARNING: wandb not installed")
        return False

    if accelerator.is_main_process:
        wc = {
            "model": model_name,
            "n_params": n_params,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum * accelerator.num_processes,
            "steps": args.steps,
            "dataset": args.dataset,
        }
        if hasattr(config, "__dataclass_fields__"):
            for f in config.__dataclass_fields__:
                wc[f"model_{f}"] = getattr(config, f)
        if hasattr(args, "model_dump"):
            for k, v in args.model_dump().items():
                if k not in wc:
                    wc[k] = v

        run_name = args.wandb_run or f"{model_name}-{n_params // 1_000_000}M"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=args.wandb_id,
            config=wc,
            resume="allow",
        )
        accelerator.print(f"[wandb] project='{args.wandb_project}' run='{run_name}'")
    return True


def log_wandb(accelerator, metrics, step, active):
    if active and accelerator.is_main_process:
        wandb.log(metrics, step=step)


def finish_wandb(accelerator, active):
    if active and accelerator.is_main_process:
        wandb.finish()
