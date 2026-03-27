"""
Baseline Transformer (GPT-style decoder-only) with optional Hyper-Connections.

Hyper-connection modes (--hc_type):
  none              – standard residual connections (vanilla transformer)
  cayley            – JPmHC: Stiefel manifold via Cayley transform
  sinkhorn          – mHC: bistochastic manifold via Sinkhorn-Knopp iteration
  fixed_rotation    – global learned Givens rotation residual (experimental)
  adaptive_rotation – data-dependent Givens rotation residual (experimental)
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hyperconnections import create_hyper_connection


@dataclass
class BaselineConfig:
    vocab_size: int = 32000
    d_model: int = 1664
    n_heads: int = 32
    n_layers: int = 16
    d_ff: int = 4096
    max_seq_len: int = 2048

    rope_theta: float = 10000.0
    gradient_checkpointing: bool = True
    use_sdpa: bool = True
    tie_embeddings: bool = False

    # Hyper-connections
    hc_type: str = "none"
    hc_num_streams: int = 4
    hc_sinkhorn_tau: float = 1.0
    hc_sinkhorn_iters: int = 20
    hc_cayley_alpha: float = 0.1
    hc_cayley_iters: int = 2

    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ── RoPE ──────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device, q.dtype)
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        def _rot(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (_rot(q) * sin), (k * cos) + (_rot(k) * sin)


# ── Attention ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.use_sdpa = cfg.use_sdpa
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(cfg.d_head, cfg.rope_theta, cfg.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            y = torch.matmul(F.softmax(att + mask, dim=-1), v)

        return self.out(y.transpose(1, 2).contiguous().view(B, T, D))


# ── MLP (SwiGLU) ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Transformer Block ────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.use_hc = cfg.hc_type != "none"

        if self.use_hc:
            hc_kw = dict(
                num_streams=cfg.hc_num_streams,
                sinkhorn_tau=cfg.hc_sinkhorn_tau,
                sinkhorn_iters=cfg.hc_sinkhorn_iters,
                cayley_alpha=cfg.hc_cayley_alpha,
                cayley_iters=cfg.hc_cayley_iters,
            )
            self.hc_attn = create_hyper_connection(cfg.hc_type, cfg.d_model, **hc_kw)
            self.hc_mlp = create_hyper_connection(cfg.hc_type, cfg.d_model, **hc_kw)

    def forward(self, x_or_streams: torch.Tensor) -> torch.Tensor:
        if self.use_hc:
            streams = self.hc_attn(x_or_streams, lambda h: self.attn(self.ln1(h)))
            streams = self.hc_mlp(streams, lambda h: self.mlp(self.ln2(h)))
            return streams
        x = x_or_streams
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full Model ────────────────────────────────────────────────────────────

class BaselineLM(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.use_hc = cfg.hc_type != "none"

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        if self.use_hc:
            # Per-stream additive bias for initial stream diversity.
            # Sinkhorn requires nonzero init to break doubly-stochastic degeneracy;
            # Cayley / Rotation work fine from zero but the bias is harmless.
            self.stream_init = nn.Parameter(torch.zeros(cfg.hc_num_streams, cfg.d_model))
            if cfg.hc_type == "sinkhorn":
                nn.init.normal_(self.stream_init, std=0.02)

        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _extract_repr(self, h: torch.Tensor) -> torch.Tensor:
        """Extract a [B, T, D] representation for stats (stream 0 for HC)."""
        if self.use_hc and h.ndim == 4:
            return h[:, :, 0, :]
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.embed(input_ids)  # [B, T, D]

        collect = getattr(self, "collect_block_stats", False)
        prev_repr: Optional[torch.Tensor] = None
        block_stats: Dict[str, float] = {}

        if self.use_hc:
            streams = x.unsqueeze(2) + self.stream_init
            for i, block in enumerate(self.blocks):
                if collect:
                    r = self._extract_repr(streams).float().detach()
                    block_stats[f"block/L{i}/input_rms"] = r.pow(2).mean().sqrt().item()
                    if prev_repr is not None:
                        block_stats[f"block/L{i}/cosine_sim_prev"] = (
                            F.cosine_similarity(prev_repr, r, dim=-1).mean().item()
                        )
                    prev_repr = r
                if self.gradient_checkpointing and self.training:
                    streams = checkpoint(block, streams, use_reentrant=False)
                else:
                    streams = block(streams)
            x = streams.mean(dim=2)
        else:
            for i, block in enumerate(self.blocks):
                if collect:
                    r = x.float().detach()
                    block_stats[f"block/L{i}/input_rms"] = r.pow(2).mean().sqrt().item()
                    if prev_repr is not None:
                        block_stats[f"block/L{i}/cosine_sim_prev"] = (
                            F.cosine_similarity(prev_repr, r, dim=-1).mean().item()
                        )
                    prev_repr = r
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        if collect:
            r = (x if not self.use_hc else x).float().detach()
            block_stats[f"block/final/output_rms"] = r.pow(2).mean().sqrt().item()
            if prev_repr is not None:
                block_stats[f"block/final/cosine_sim_prev"] = (
                    F.cosine_similarity(prev_repr, r, dim=-1).mean().item()
                )
            cos_vals = [v for k, v in block_stats.items() if "cosine_sim" in k]
            if cos_vals:
                block_stats["block/mean_cosine_sim"] = sum(cos_vals) / len(cos_vals)
                block_stats["block/max_cosine_sim"] = max(cos_vals)
            self.last_block_stats = block_stats

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.cfg.max_seq_len
                else input_ids[:, -self.cfg.max_seq_len :]
            )
            logits = self(idx_cond)["logits"][:, -1, :]

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cum_probs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if self.cfg.eos_token_id is not None and (next_token == self.cfg.eos_token_id).any():
                break

        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
