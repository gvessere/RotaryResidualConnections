"""
Hyper-Connection implementations for multi-stream residual mixing.

Four variants (all share the mHC paper's data-dependent width/depth structure):
  1. SinkhornHyperConnection          – H_res via log-domain Sinkhorn-Knopp     (mHC)
  2. CayleyHyperConnection            – H_res via iterative Cayley transform    (JPmHC)
  3. FixedRotationHyperConnection     – H_res via global Givens rotation        (experimental)
  4. AdaptiveRotationHyperConnection  – H_res via data-dependent Givens         (experimental)

Update rule per the mHC paper (arXiv:2512.24880, Eq 3, Section 4.2):

    x_{l+1} = H_res @ x_l  +  H_post^T ⊙ F(H_pre · x_l, W_l)

    H_pre  = σ(H̃_pre)              non-negative, data-dependent     (Eq 8)
    H_post = 2σ(H̃_post)            non-negative, data-dependent     (Eq 8)
    H_res  = Sinkhorn-Knopp(H̃_res)  doubly-stochastic               (Eq 8-9)

Dynamic + static parameterisation follows the mHC paper Eq 7:
    H̃ = α · (x̃'_l · φ) + b       (no tanh — mHC drops it from original HC)

Diagnostics: set module.collect_stats = True to populate module.last_stats
with per-layer metrics.  Use compute_composite_h_res_stats() for cross-layer
composite mapping analysis (Amax Gain Magnitude, Section 3.1 / 5.4).

References:
  - Sinkhorn mHC:       arXiv:2512.24880
  - Cayley JPmHC:       arXiv:2602.18308
  - Original HC:        arXiv:2409.19606
  - Frac-Connections:   arXiv:2503.14125
"""

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_hyper_connection(hc_type: str, hidden_size: int, **kwargs) -> nn.Module:
    """Factory for hyper-connection modules."""
    constructors = {
        "cayley": CayleyHyperConnection,
        "sinkhorn": SinkhornHyperConnection,
        "fixed_rotation": FixedRotationHyperConnection,
        "adaptive_rotation": AdaptiveRotationHyperConnection,
    }
    if hc_type not in constructors:
        raise ValueError(f"Unknown hc_type '{hc_type}'. Choose from {list(constructors)}")
    return constructors[hc_type](hidden_size, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinkhorn_log(
    logits: torch.Tensor,
    num_iters: int = 20,
    tau: float = 1.0,
) -> torch.Tensor:
    """Project logits onto the doubly-stochastic manifold (Birkhoff polytope).

    Uses log-domain Sinkhorn-Knopp for numerical stability.
    Handles both 2-D [n, n] and batched [..., n, n] inputs.

    The mHC paper (Eq 9) uses plain Sinkhorn without temperature (tau=1.0)
    and t_max=20 iterations (Table 5).
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_mu = -math.log(n)

    u = Z.new_zeros(Z.shape[:-1])
    v = Z.new_zeros(Z.shape[:-1])

    for _ in range(num_iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_mu - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2)) * n


# ---------------------------------------------------------------------------
# Base class — shared H_pre / H_post / forward logic
# ---------------------------------------------------------------------------

class _HyperConnectionBase(nn.Module):
    """Shared structure for all hyper-connection variants.

    Uses raw nn.Parameter + F.linear (instead of nn.Linear) so that
    model-level _init_weights hooks (which re-init all nn.Linear to
    std=0.02) cannot overwrite the critical zero/identity initialisation.

    Set self.collect_stats = True to populate self.last_stats with
    per-forward diagnostics (zero overhead when disabled).
    """

    def __init__(self, hidden_size: int, num_streams: int = 4, **_kwargs):
        super().__init__()
        n = self.num_streams = num_streams
        self.hidden_size = hidden_size
        self.norm = nn.RMSNorm(hidden_size)

        # H_pre (width): σ(·), data-dependent (Eq 8)
        # Static init: near one-hot on stream 0  →  σ(8)≈1, σ(-8)≈0
        static_pre = torch.full((n,), -8.0)
        static_pre[0] = 8.0
        self.static_pre = nn.Parameter(static_pre)
        self._dyn_pre_w = nn.Parameter(torch.zeros(n, hidden_size))
        self._dyn_pre_scale = nn.Parameter(torch.tensor(1e-2))

        # H_post (depth): 2σ(·), data-dependent (Eq 8)
        # Static init: 2σ(0) = 1.0 per stream → mean = 1.0, matching standard residual
        self.static_post = nn.Parameter(torch.zeros(n))
        self._dyn_post_w = nn.Parameter(torch.zeros(n, hidden_size))
        self._dyn_post_scale = nn.Parameter(torch.tensor(1e-2))

        self.last_stats: Dict[str, float] = {}

    def _compute_h_res(self, normed: torch.Tensor) -> torch.Tensor:
        """Return H_res given normed hidden state.  Shape: [..., n, n] or [n, n]."""
        raise NotImplementedError

    def _collect_variant_stats(
        self, stats: Dict[str, float], h_res: torch.Tensor
    ) -> None:
        """Override in subclasses to add variant-specific diagnostics."""

    def forward(self, streams: torch.Tensor, sublayer_fn: Callable) -> torch.Tensor:
        B, T, n, D = streams.shape

        x_avg = streams.mean(dim=2)
        normed = self.norm(x_avg).float()

        # H_pre: σ(α·(x̃'·φ) + b)  — non-negative, data-dependent (Eq 7-8)
        dyn_pre = F.linear(normed, self._dyn_pre_w) * self._dyn_pre_scale
        h_pre = torch.sigmoid(self.static_pre + dyn_pre)

        x_in = torch.einsum("btn, btnd -> btd", h_pre.to(streams.dtype), streams)
        y = sublayer_fn(x_in)

        # H_post: 2σ(α·(x̃'·φ) + b)  — non-negative, data-dependent (Eq 8)
        dyn_post = F.linear(normed, self._dyn_post_w) * self._dyn_post_scale
        h_post = 2.0 * torch.sigmoid(self.static_post + dyn_post)

        # H_res: variant-specific constrained matrix (computed in fp32)
        h_res = self._compute_h_res(normed)

        s_res = torch.matmul(h_res.to(streams.dtype), streams)
        y_post = h_post.to(streams.dtype).unsqueeze(-1) * y.unsqueeze(2)

        out = s_res + y_post

        if getattr(self, "collect_stats", False):
            self._collect_all_stats(h_pre, h_post, h_res)

        return out

    @torch.no_grad()
    def _collect_all_stats(
        self,
        h_pre: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> None:
        stats: Dict[str, float] = {}

        # ── H_pre diagnostics ────────────────────────────────────────
        stats["h_pre/min"] = h_pre.min().item()
        stats["h_pre/max"] = h_pre.max().item()
        stats["h_pre/mean"] = h_pre.mean().item()
        p = h_pre / (h_pre.sum(dim=-1, keepdim=True) + 1e-8)
        stats["h_pre/entropy"] = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()

        # ── H_post diagnostics ───────────────────────────────────────
        stats["h_post/min"] = h_post.min().item()
        stats["h_post/max"] = h_post.max().item()
        stats["h_post/mean"] = h_post.mean().item()
        stats["h_post/sum"] = h_post.sum(dim=-1).mean().item()

        # ── Dynamic scale values ─────────────────────────────────────
        stats["scale/pre"] = self._dyn_pre_scale.item()
        stats["scale/post"] = self._dyn_post_scale.item()

        # ── H_res diagnostics (Amax Gain Magnitude — §3.1, §5.4) ────
        hr = h_res.float()
        if hr.dim() > 2:
            # [B, T, n, n] → per-token gain, then average
            fwd = hr.abs().sum(dim=-1).max(dim=-1).values
            bwd = hr.abs().sum(dim=-2).max(dim=-1).values
            stats["h_res/fwd_gain"] = fwd.mean().item()
            stats["h_res/bwd_gain"] = bwd.mean().item()

            hr_avg = hr.mean(dim=(0, 1))
        else:
            stats["h_res/fwd_gain"] = hr.abs().sum(dim=-1).max().item()
            stats["h_res/bwd_gain"] = hr.abs().sum(dim=-2).max().item()
            hr_avg = hr

        I = torch.eye(self.num_streams, device=hr_avg.device, dtype=hr_avg.dtype)
        stats["h_res/dist_from_I"] = (hr_avg - I).norm().item()
        stats["h_res/min"] = hr.min().item()
        stats["h_res/max"] = hr.max().item()

        self._collect_variant_stats(stats, h_res)

        # Store the batch-averaged H_res for composite analysis
        self._last_h_res_avg = hr_avg.detach()
        self.last_stats = stats


# ---------------------------------------------------------------------------
# Cross-layer composite mapping diagnostics
# ---------------------------------------------------------------------------

def compute_composite_h_res_stats(
    hc_modules: List[nn.Module],
) -> Dict[str, float]:
    """Compute stability metrics for the composite residual mapping Π = ∏ H_res.

    Call after a forward pass with collect_stats=True on each module.
    Uses the batch-averaged H_res stored by each module.

    Returns dict with:
        composite/fwd_gain   – max absolute row sum (forward amplification)
        composite/bwd_gain   – max absolute column sum (backward amplification)
        composite/dist_from_I – Frobenius distance from identity
        composite/spectral_norm – largest singular value
    """
    composite: Optional[torch.Tensor] = None
    for mod in hc_modules:
        hr = getattr(mod, "_last_h_res_avg", None)
        if hr is None:
            continue
        composite = hr if composite is None else hr @ composite

    if composite is None:
        return {}

    I = torch.eye(composite.shape[-1], device=composite.device, dtype=composite.dtype)
    return {
        "composite/fwd_gain": composite.abs().sum(dim=-1).max().item(),
        "composite/bwd_gain": composite.abs().sum(dim=-2).max().item(),
        "composite/dist_from_I": (composite - I).norm().item(),
        "composite/spectral_norm": torch.linalg.svdvals(composite)[0].item(),
    }


# ---------------------------------------------------------------------------
# 1. Sinkhorn — Doubly-stochastic manifold (mHC, arXiv:2512.24880)
# ---------------------------------------------------------------------------

class SinkhornHyperConnection(_HyperConnectionBase):
    """H_res projected onto the Birkhoff polytope via log-domain Sinkhorn-Knopp.

    Paper defaults (Table 5): tau=1.0 (no temperature), t_max=20 iterations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        sinkhorn_tau: float = 1.0,
        **_kwargs,
    ):
        super().__init__(hidden_size, num_streams, **_kwargs)
        n = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau

        static_res = torch.full((n, n), -8.0)
        static_res.fill_diagonal_(0.0)
        self.static_res = nn.Parameter(static_res)
        self._dyn_res_w = nn.Parameter(torch.zeros(n * n, hidden_size))
        self._dyn_res_scale = nn.Parameter(torch.tensor(1e-2))

    def _compute_h_res(self, normed: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        dyn = F.linear(normed, self._dyn_res_w) * self._dyn_res_scale
        logits = self.static_res + dyn.unflatten(-1, (n, n))
        return sinkhorn_log(logits, self.sinkhorn_iters, self.sinkhorn_tau)

    def _collect_variant_stats(
        self, stats: Dict[str, float], h_res: torch.Tensor
    ) -> None:
        hr = h_res.float()
        if hr.dim() > 2:
            hr = hr.mean(dim=(0, 1))
        row_sums = hr.sum(dim=-1)
        col_sums = hr.sum(dim=-2)
        stats["h_res/row_sum_err"] = (row_sums - 1).abs().max().item()
        stats["h_res/col_sum_err"] = (col_sums - 1).abs().max().item()
        stats["scale/res"] = self._dyn_res_scale.item()


# ---------------------------------------------------------------------------
# 2. Cayley — Stiefel manifold (JPmHC, arXiv:2602.18308)
# ---------------------------------------------------------------------------

class CayleyHyperConnection(_HyperConnectionBase):
    """H_res projected onto the orthogonal group via iterative Cayley transform.

    Raw logits are antisymmetrised (W = A - A^T) before the Cayley map,
    so initialising static_res = 0 gives W = 0 → H_res = I.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        cayley_alpha: float = 0.1,
        cayley_iters: int = 2,
        **_kwargs,
    ):
        super().__init__(hidden_size, num_streams, **_kwargs)
        n = num_streams
        self.cayley_alpha = cayley_alpha
        self.cayley_iters = cayley_iters

        self.static_res = nn.Parameter(torch.zeros(n, n))
        self._dyn_res_w = nn.Parameter(torch.zeros(n * n, hidden_size))
        self._dyn_res_scale = nn.Parameter(torch.tensor(1e-2))
        self.register_buffer("_I", torch.eye(n), persistent=False)

    def _compute_h_res(self, normed: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        dyn = F.linear(normed, self._dyn_res_w) * self._dyn_res_scale
        raw = self.static_res + dyn.unflatten(-1, (n, n))

        W = raw - raw.transpose(-1, -2)
        I = self._I.to(dtype=W.dtype)
        Y = I + self.cayley_alpha * W
        for _ in range(self.cayley_iters):
            Y = I + 0.5 * self.cayley_alpha * torch.matmul(W, I + Y)
        return Y

    def _collect_variant_stats(
        self, stats: Dict[str, float], h_res: torch.Tensor
    ) -> None:
        hr = h_res.float()
        if hr.dim() > 2:
            hr = hr.mean(dim=(0, 1))
        orth_err = (hr @ hr.transpose(-1, -2)) - torch.eye(
            self.num_streams, device=hr.device, dtype=hr.dtype
        )
        stats["h_res/orth_err"] = orth_err.norm().item()
        stats["scale/res"] = self._dyn_res_scale.item()


# ---------------------------------------------------------------------------
# 3. FixedRotation — Global learned Givens rotation (experimental)
# ---------------------------------------------------------------------------

class FixedRotationHyperConnection(_HyperConnectionBase):
    """H_res is a single global SO(n) matrix built from n(n-1)/2 Givens rotations.

    The rotation does not depend on input (same for every token/batch).
    Angles are initialised near zero → identity residual at the start of training.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        angle_init_std: float = 0.01,
        **_kwargs,
    ):
        super().__init__(hidden_size, num_streams, **_kwargs)
        n = num_streams
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.register_buffer("_pairs", torch.tensor(pairs, dtype=torch.long), persistent=False)
        self.angles = nn.Parameter(torch.randn(len(pairs)) * angle_init_std)
        self._cached_R: torch.Tensor | None = None

    def _build_rotation(self) -> torch.Tensor:
        if not self.training and self._cached_R is not None:
            return self._cached_R

        n = self.num_streams
        num_pairs = self._pairs.shape[0]
        c = torch.cos(self.angles)
        s = torch.sin(self.angles)

        eye = torch.eye(n, dtype=c.dtype, device=c.device)
        G = eye.unsqueeze(0).expand(num_pairs, -1, -1).clone()
        idx = torch.arange(num_pairs, device=c.device)
        pi, pj = self._pairs[:, 0], self._pairs[:, 1]
        G[idx, pi, pi] = c
        G[idx, pi, pj] = -s
        G[idx, pj, pi] = s
        G[idx, pj, pj] = c

        R = G[0]
        for k in range(1, num_pairs):
            R = G[k] @ R

        if not self.training:
            self._cached_R = R
        return R

    def train(self, mode: bool = True):
        if mode:
            self._cached_R = None
        return super().train(mode)

    def _compute_h_res(self, normed: torch.Tensor) -> torch.Tensor:
        return self._build_rotation()

    def _collect_variant_stats(
        self, stats: Dict[str, float], h_res: torch.Tensor
    ) -> None:
        stats["rotation/angle_mean"] = self.angles.abs().mean().item()
        stats["rotation/angle_max"] = self.angles.abs().max().item()
        hr = h_res.float()
        orth_err = (hr @ hr.T) - torch.eye(
            self.num_streams, device=hr.device, dtype=hr.dtype
        )
        stats["h_res/orth_err"] = orth_err.norm().item()


# ---------------------------------------------------------------------------
# 4. AdaptiveRotation — Data-dependent Givens rotation (experimental)
# ---------------------------------------------------------------------------

class AdaptiveRotationHyperConnection(_HyperConnectionBase):
    """H_res is a per-token SO(n) matrix built from data-dependent Givens angles.

    Angles are predicted from the stream-averaged hidden state.
    Small-init projection ensures the rotation starts near identity.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        angle_init_std: float = 0.01,
        **_kwargs,
    ):
        super().__init__(hidden_size, num_streams, **_kwargs)
        n = num_streams
        num_pairs = n * (n - 1) // 2
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.register_buffer("_pairs", torch.tensor(pairs, dtype=torch.long), persistent=False)

        self._angle_w = nn.Parameter(
            torch.randn(num_pairs, hidden_size) * (angle_init_std / math.sqrt(hidden_size))
        )
        self._angle_b = nn.Parameter(torch.zeros(num_pairs))

    def _build_rotation(self, angles: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        c = torch.cos(angles)
        s = torch.sin(angles)

        eye = torch.eye(n, dtype=c.dtype, device=c.device)
        R = eye.expand(*angles.shape[:-1], n, n).clone()

        for k in range(self._pairs.shape[0]):
            i = self._pairs[k, 0].item()
            j = self._pairs[k, 1].item()
            G = eye.expand_as(R).clone()
            G[..., i, i] = c[..., k]
            G[..., i, j] = -s[..., k]
            G[..., j, i] = s[..., k]
            G[..., j, j] = c[..., k]
            R = G @ R

        return R

    def _compute_h_res(self, normed: torch.Tensor) -> torch.Tensor:
        angles = F.linear(normed, self._angle_w, self._angle_b)
        return self._build_rotation(angles)

    def _collect_variant_stats(
        self, stats: Dict[str, float], h_res: torch.Tensor
    ) -> None:
        hr = h_res.float()
        if hr.dim() > 2:
            hr_avg = hr.mean(dim=(0, 1))
        else:
            hr_avg = hr
        orth_err = (hr_avg @ hr_avg.T) - torch.eye(
            self.num_streams, device=hr_avg.device, dtype=hr_avg.dtype
        )
        stats["h_res/orth_err"] = orth_err.norm().item()
