# HyperBaseline

GPT-style decoder-only transformer with three optional **hyper-connection** modes for multi-stream residual mixing, trained on The Pile. Experiment parameters are managed with [Hydra](https://hydra.cc/).

## Hyper-Connection Modes

| `arch=` | Method | Connection Matrices | Reference |
|---|---|---|---|
| `vanilla` | Standard residual (`x + sublayer(x)`) | — | — |
| `cayley` | JPmHC – Stiefel manifold via iterative Cayley transform | Learned, input-dependent; residual constrained to orthogonal | [arXiv:2602.18308](https://arxiv.org/abs/2602.18308) |
| `sinkhorn` | mHC – Bistochastic manifold via Sinkhorn-Knopp iteration | Learned, input-dependent; all matrices doubly-stochastic | [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) |
| `rotation` | Fixed Givens rotation matrices | Non-learnable; angle θ = π/(2n) from number of streams | Experimental |

All HC modes maintain **persistent multi-stream state** `[B, T, n, D]` throughout the network. Streams are initialised from the token embedding plus a per-stream learnable bias and collapsed via mean-pooling before the LM head.

## Optimizers

| `use_muon=` | Linear (2-D) weights | Other params | Backend |
|---|---|---|---|
| `false` (default) | AdamW | AdamW | DeepSpeed ZeRO-3 |
| `true` | **Muon** (Newton-Schulz orthogonalised SGD) | AdamW | Accelerate DDP |

When `use_muon=true`, DeepSpeed is disabled and the Muon optimizer handles all 2-D (Linear weight) parameters while an internal AdamW handles biases, norms, and embeddings. LR follows a cosine schedule with warmup.

## Quick Start

```bash
pip install -r requirements.txt
```

### Single-GPU (argparse, for debugging)

```bash
python train.py --hc_type cayley --hc_num_streams 4
```

### Multi-GPU with Hydra (recommended)

```bash
# Vanilla baseline, DeepSpeed ZeRO-3
accelerate launch --num_processes 2 train_zero3.py

# Cayley HC
accelerate launch --num_processes 2 train_zero3.py arch=cayley

# Sinkhorn HC, 8 streams
accelerate launch --num_processes 2 train_zero3.py arch=sinkhorn arch.hc_num_streams=8

# Fixed rotation
accelerate launch --num_processes 2 train_zero3.py arch=rotation

# Muon optimizer + Cayley HC
accelerate launch --num_processes 2 train_zero3.py use_muon=true arch=cayley

# Override LR, enable W&B
accelerate launch --num_processes 2 train_zero3.py lr=1e-4 wandb=true wandb_run=my-run
```

### Generation

```bash
python test_generate.py --checkpoint checkpoints/baseline_cayley_deepspeed_500.pt
```

## Hydra Config Structure

```
config/
├── cfg_train.yaml          # Main config (data, training, optimizer, logging)
└── arch/
    ├── vanilla.yaml        # hc_type: none
    ├── cayley.yaml         # hc_type: cayley
    ├── sinkhorn.yaml       # hc_type: sinkhorn
    └── rotation.yaml       # hc_type: rotation
```

Override any value from the command line with Hydra syntax (`key=value`). The resolved config is saved to `checkpoints/config.yaml` on each run.

## Architecture Defaults

| Parameter | Value |
|---|---|
| `d_model` | 1664 |
| `n_heads` | 32 |
| `n_layers` | 16 |
| `d_ff` | 4096 |
| `max_seq_len` | 2048 |

With `hc_num_streams=4`, activation memory per block is ~4× the vanilla baseline. Gradient checkpointing is enabled by default.

## Files

| File | Purpose |
|---|---|
| `model.py` | `BaselineConfig`, `BaselineLM`, transformer blocks |
| `hyperconnections.py` | `CayleyHyperConnection`, `SinkhornHyperConnection`, `FixedRotationHyperConnection`, `AdaptiveRotationHyperConnection` |
| `muon.py` | Muon optimizer (Newton-Schulz orthogonalisation for Linear weights) |
| `data.py` | Pile dataset loading, tokenisation, DataLoaders |
| `train.py` | Single-GPU training with AMP (argparse) |
| `train_zero3.py` | Multi-GPU Accelerate + DeepSpeed/DDP (Hydra config) |
| `train_utils.py` | Checkpoints, W&B, resume helpers |
| `test_generate.py` | Inference / text generation |
| `config/` | Hydra YAML configs |
| `ds/zero3_fp16.json` | DeepSpeed configuration |
