"""
Hyper-Connection implementations for multi-stream residual mixing.

Three variants:
  1. CayleyHyperConnection   – Stiefel manifold via iterative Cayley transform  (JPmHC)
  2. SinkhornHyperConnection – Bistochastic manifold via Sinkhorn-Knopp          (mHC)
  3. RotationHyperConnection – Fixed Givens rotation matrices                    (experimental)

All operate on persistent multi-stream state [B, T, n, D]:
    forward(streams, sublayer_fn) -> new_streams

References:
  - Cayley:   arXiv:2602.18308  (JPmHC – Dynamical Isometry via Orthogonal HC)
  - Sinkhorn: arXiv:2512.24880  (mHC – Manifold-Constrained Hyper-Connections)
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_hyper_connection(hc_type: str, hidden_size: int, **kwargs) -> nn.Module:
    """Factory for hyper-connection modules."""
    constructors = {
        "cayley": CayleyHyperConnection,
        "sinkhorn": SinkhornHyperConnection,
        "rotation": RotationHyperConnection,
    }
    if hc_type not in constructors:
        raise ValueError(f"Unknown hc_type '{hc_type}'. Choose from {list(constructors)}")
    return constructors[hc_type](hidden_size, **kwargs)


# ---------------------------------------------------------------------------
# 1. Cayley – Stiefel manifold (JPmHC, arXiv:2602.18308)
# ---------------------------------------------------------------------------

class CayleyHyperConnection(nn.Module):
    """
    Pre-connection:  row-stochastic   (softmax over cols)
    Post-connection: column-stochastic (softmax over rows)
    Residual:        orthogonal        (iterative Cayley transform)
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        tau: float = 1.0,
        cayley_alpha: float = 0.1,
        cayley_iters: int = 2,
        **_kwargs,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.tau = tau
        self.cayley_alpha = cayley_alpha
        self.cayley_iters = cayley_iters

        self.norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 3 * num_streams * num_streams, bias=True)
        self.register_buffer("_I", torch.eye(num_streams), persistent=False)

    def _iterative_cayley(self, raw: torch.Tensor) -> torch.Tensor:
        W = raw - raw.transpose(-1, -2)
        I = self._I.to(dtype=W.dtype, device=W.device)
        Y = I + self.cayley_alpha * W
        for _ in range(self.cayley_iters):
            Y = I + 0.5 * self.cayley_alpha * torch.matmul(W, I + Y)
        return Y

    def forward(self, streams: torch.Tensor, sublayer_fn: Callable) -> torch.Tensor:
        B, T, n, D = streams.shape

        x_avg = streams.mean(dim=2)
        gates = self.gate_proj(self.norm(x_avg.float()).to(streams.dtype))
        pre_raw, post_raw, res_raw = gates.chunk(3, dim=-1)

        pre_raw = pre_raw.view(B, T, n, n).float()
        post_raw = post_raw.view(B, T, n, n).float()
        res_raw = res_raw.view(B, T, n, n).float()

        h_pre = torch.softmax(pre_raw / self.tau, dim=-1)
        h_post = torch.softmax(post_raw / self.tau, dim=-2)
        h_res = self._iterative_cayley(res_raw)

        x_pre = torch.einsum("btij,btjd->btid", h_pre.to(streams.dtype), streams)
        x_in = x_pre.mean(dim=2)

        y = sublayer_fn(x_in)

        y_exp = y.unsqueeze(2).expand_as(streams)
        y_post = torch.einsum("btij,btjd->btid", h_post.to(streams.dtype), y_exp)
        s_res = torch.einsum("btij,btjd->btid", h_res.to(streams.dtype), streams)

        return s_res + y_post


# ---------------------------------------------------------------------------
# 2. Sinkhorn – Bistochastic manifold (mHC, arXiv:2512.24880)
# ---------------------------------------------------------------------------

class SinkhornHyperConnection(nn.Module):
    """
    All three connection matrices are projected onto the doubly-stochastic
    manifold via alternating row/column normalisation (Sinkhorn-Knopp),
    restoring the identity-mapping property of residual connections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        tau: float = 1.0,
        sinkhorn_iters: int = 5,
        **_kwargs,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters

        self.norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 3 * num_streams * num_streams, bias=True)

    def _sinkhorn_knopp(self, log_alpha: torch.Tensor) -> torch.Tensor:
        M = torch.exp(log_alpha / self.tau)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M

    def forward(self, streams: torch.Tensor, sublayer_fn: Callable) -> torch.Tensor:
        B, T, n, D = streams.shape

        x_avg = streams.mean(dim=2)
        gates = self.gate_proj(self.norm(x_avg.float()).to(streams.dtype))
        pre_raw, post_raw, res_raw = gates.chunk(3, dim=-1)

        pre_raw = pre_raw.view(B, T, n, n).float()
        post_raw = post_raw.view(B, T, n, n).float()
        res_raw = res_raw.view(B, T, n, n).float()

        h_pre = self._sinkhorn_knopp(pre_raw)
        h_post = self._sinkhorn_knopp(post_raw)
        h_res = self._sinkhorn_knopp(res_raw)

        x_pre = torch.einsum("btij,btjd->btid", h_pre.to(streams.dtype), streams)
        x_in = x_pre.mean(dim=2)

        y = sublayer_fn(x_in)

        y_exp = y.unsqueeze(2).expand_as(streams)
        y_post = torch.einsum("btij,btjd->btid", h_post.to(streams.dtype), y_exp)
        s_res = torch.einsum("btij,btjd->btid", h_res.to(streams.dtype), streams)

        return s_res + y_post


# ---------------------------------------------------------------------------
# 3. Rotation – Fixed Givens rotation matrices (experimental)
# ---------------------------------------------------------------------------

class RotationHyperConnection(nn.Module):
    """
    Hard-coded orthogonal mixing via paired Givens rotations.

    Angle theta = pi / (2 * num_streams) ensures mild, geometry-aware
    cross-stream coupling that grows more conservative with more streams.
    Pre uses R, post uses R^T (inverse), residual uses R.

    No learnable parameters in the connection matrices; diversity across
    streams emerges from differing row-sums of the orthogonal matrix.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        **_kwargs,
    ):
        super().__init__()
        self.num_streams = num_streams

        theta = math.pi / (2.0 * num_streams)
        R = self._build_givens_rotation(num_streams, theta)
        self.register_buffer("R_fwd", R, persistent=False)
        self.register_buffer("R_inv", R.t().contiguous(), persistent=False)

    @staticmethod
    def _build_givens_rotation(n: int, theta: float) -> torch.Tensor:
        R = torch.eye(n, dtype=torch.float32)
        c, s = math.cos(theta), math.sin(theta)
        for i in range(0, n - 1, 2):
            R[i, i] = c
            R[i, i + 1] = -s
            R[i + 1, i] = s
            R[i + 1, i + 1] = c
        return R

    def forward(self, streams: torch.Tensor, sublayer_fn: Callable) -> torch.Tensor:
        dtype = streams.dtype
        R = self.R_fwd.to(dtype)
        R_inv = self.R_inv.to(dtype)

        x_pre = torch.einsum("ij,btjd->btid", R, streams)
        x_in = x_pre.mean(dim=2)

        y = sublayer_fn(x_in)

        y_exp = y.unsqueeze(2).expand_as(streams)
        y_post = torch.einsum("ij,btjd->btid", R_inv, y_exp)
        s_res = torch.einsum("ij,btjd->btid", R, streams)

        return s_res + y_post
