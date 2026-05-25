"""Minimal Muon optimizer vendored from Keller Jordan's implementation.

Muon = "Momentum-Orthogonalized" — applies a Newton-Schulz iteration to the
momentum buffer of each 2D weight to whiten it before the update step.
Designed for 2D matrices only; non-2D params should be routed to AdamW.

Reference: https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import torch


def _newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """5-step quintic Newton-Schulz to orthogonalize a 2D matrix in bf16."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices only."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    # Muon group should only contain 2D params; skip defensively.
                    continue

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                g = _newton_schulz5(g, steps=ns_steps)
                # Scale by sqrt(max(dim_out/dim_in, 1)) so update RMS ~ matches AdamW.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr * scale)

        return loss


class CompositeOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers to look like a single one.

    Exposes a flat ``param_groups`` so LambdaLR can scale all groups by the
    same lr_lambda. Each optimizer's groups appear in order. Subclasses
    torch.optim.Optimizer for ``isinstance`` checks but does not call
    super().__init__ since state is delegated to the inner optimizers.
    """

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
        self.defaults = {}
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {f"opt{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, sd):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(sd[f"opt{i}"])
