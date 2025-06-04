"""Optimiser and per-iteration LR schedule (linear warmup followed by cosine or step decay)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch


def build_optimizer(param_groups: List[Dict], cfg: Dict) -> torch.optim.Optimizer:
    lr = float(cfg.get("lr", 0.01))
    wd = float(cfg.get("weight_decay", 1e-4))
    groups = [{"params": g["params"], "lr": lr * g.get("lr_mult", 1.0), "lr_mult": g.get("lr_mult", 1.0)} for g in param_groups]
    name = cfg.get("optimizer", "sgd").lower()
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name!r}")


class WarmupScheduler:
    """Stepped once per iteration. ``base_lr * lr_mult`` scaled by warmup and decay factors."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        warmup_iters: int = 500,
        warmup_factor: float = 0.001,
        mode: str = "cosine",
        milestones: Sequence[float] = (0.67, 0.89),
        gamma: float = 0.1,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.total_iters = max(1, total_iters)
        self.warmup_iters = min(warmup_iters, self.total_iters - 1) if self.total_iters > 1 else 0
        self.warmup_factor = warmup_factor
        self.mode = mode
        self.milestones = [int(m * self.total_iters) for m in milestones]
        self.gamma = gamma
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.iter = 0
        self._apply()

    def factor(self, it: int) -> float:
        if it < self.warmup_iters:
            alpha = it / max(1, self.warmup_iters)
            return self.warmup_factor * (1 - alpha) + alpha
        if self.mode == "cosine":
            progress = (it - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
            progress = min(1.0, progress)
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if self.mode == "step":
            return self.gamma ** sum(it >= m for m in self.milestones)
        return 1.0

    def _apply(self) -> None:
        f = self.factor(self.iter)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * f

    def step(self) -> None:
        self.iter += 1
        self._apply()

    def state_dict(self) -> Dict:
        return {"iter": self.iter}

    def load_state_dict(self, state: Dict) -> None:
        self.iter = state["iter"]
        self._apply()
