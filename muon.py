"""
Muon optimizer – MomentUm Orthogonalized by Newton-Schulz.

Applies Newton-Schulz orthogonalization to gradient updates for 2D (Linear
weight) parameters.  Non-2D parameters (embeddings, biases, norms) fall back
to AdamW within the same optimizer instance.

Usage:
    muon_params = [p for p in model.parameters() if p.ndim == 2]
    adam_params  = [p for p in model.parameters() if p.ndim != 2]
    optimizer = Muon([
        {"params": muon_params, "use_muon": True,  "lr": 0.02},
        {"params": adam_params,  "use_muon": False, "lr": 3e-4,
         "weight_decay": 0.1, "adamw_betas": (0.9, 0.95)},
    ])

Reference: https://github.com/KellerJordan/Muon
           https://github.com/MoonshotAI/Moonlight  (LR adjustment)
Ported from /Users/geryvessere/Documents/src/URM/models/muon.py
"""

import math
import torch


# ── Newton-Schulz orthogonalization ──────────────────────────────────────

def _ns_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    return torch.addmm(X, B, X, alpha=1.0, beta=a)


_NS_COEFFS = [
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
]


def _msign_impl(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate matrix sign via quintic Newton-Schulz iteration."""
    if G.ndim < 2:
        raise ValueError("Input must be >= 2-D")
    if G.dtype != torch.float32:
        G = G.float()

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)

    for i in range(steps):
        a, b, c = _NS_COEFFS[min(i, len(_NS_COEFFS) - 1)]
        X = _ns_step(X, a, b, c)

    return X.mT if transpose else X


try:
    msign = torch.compile(_msign_impl)
except Exception:
    msign = _msign_impl


def _adjust_lr_for_muon(lr: float, matched_rms: float, shape: torch.Size) -> float:
    A, B = shape[:2]
    return lr * math.sqrt(max(A, B)) * matched_rms


# ── Optimizer ────────────────────────────────────────────────────────────

class Muon(torch.optim.Optimizer):
    """
    Combined Muon (2-D weights) + AdamW (everything else).

    Each param group must set ``use_muon`` to ``True`` or ``False``.
    Muon groups also accept: momentum, nesterov, ns_steps, matched_adamw_rms.
    AdamW groups also accept: adamw_betas, adamw_eps.
    Both accept: lr, weight_decay.
    """

    def __init__(
        self,
        param_groups,
        lr: float = 2e-2,
        weight_decay: float = 0.1,
        matched_adamw_rms: float = 0.2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple = (0.95, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ── Phase 1: Muon groups ─────────────────────────────────────
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            matched_rms = group["matched_adamw_rms"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "muon_buf" not in state:
                    state["muon_buf"] = torch.zeros_like(g)
                buf = state["muon_buf"]
                buf.mul_(momentum).add_(g)
                direction = g.add(buf, alpha=momentum) if nesterov else buf

                update = msign(direction.bfloat16(), steps=ns_steps)
                adj_lr = _adjust_lr_for_muon(lr, matched_rms, direction.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(update.to(p.dtype), alpha=-adj_lr)

        # ── Phase 2: AdamW groups ────────────────────────────────────
        for group in self.param_groups:
            if group.get("use_muon", False):
                continue

            if "step" not in group:
                group["step"] = 0
            group["step"] += 1
            t = group["step"]
            lr = group["lr"]
            wd = group["weight_decay"]
            b1, b2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)

                state["exp_avg"].lerp_(g, 1 - b1)
                state["exp_avg_sq"].lerp_(g.square(), 1 - b2)

                bias1 = 1 - b1**t
                bias2 = 1 - b2**t
                step_size = lr / (bias1 / bias2**0.5)

                update = state["exp_avg"] / (state["exp_avg_sq"].sqrt() + eps)
                p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-step_size)

        return loss
