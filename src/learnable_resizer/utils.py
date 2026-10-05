"""Training utilities: seeding, metrics, checkpoints, optimizer groups, FLOPs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def topk_correct(logits: Tensor, targets: Tensor, ks: tuple[int, ...] = (1, 5)) -> list[int]:
    """Number of samples whose target is within the top-k predictions, for each k."""
    pred = logits.topk(max(ks), dim=1).indices
    hits = pred.eq(targets[:, None])
    return [int(hits[:, :k].any(dim=1).sum()) for k in ks]


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Apply weight decay to conv/linear weights only, not to biases or norm parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def count_gflops(model: nn.Module, input_shape: tuple[int, ...], device: torch.device) -> float:
    """Forward FLOPs (2 x multiply-adds) of a single sample, in billions."""
    was_training = model.training
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    return counter.get_total_flops() / 1e9


def save_checkpoint(path: str | Path, **state: Any) -> None:
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_backbone_weights(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Initialize ``model.backbone`` from a previous run's checkpoint (e.g. the bilinear
    baseline), leaving the resizer at its random initialization."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    prefix = "backbone."
    backbone_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    model.backbone.load_state_dict(backbone_state, strict=True)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    step_epochs: int,
    gamma: float,
    warmup_epochs: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step schedule: optional linear warmup, then decay by ``gamma`` every
    ``step_epochs`` epochs (paper: 0.94 every 2 epochs). Call ``.step()`` every iteration."""
    warmup_steps = round(warmup_epochs * steps_per_epoch)

    def factor(step: int) -> float:
        warmup = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        return warmup * gamma ** (step // (step_epochs * steps_per_epoch))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
