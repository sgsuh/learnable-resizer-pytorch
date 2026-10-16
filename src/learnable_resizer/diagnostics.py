"""Diagnostics for eval-mode accuracy drops: BatchNorm statistics and resizer outputs."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from learnable_resizer.models.resizer import LearnableResizer


def batch_norms(module: nn.Module) -> dict[str, nn.modules.batchnorm._BatchNorm]:
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}


def _batches(loader: DataLoader, num_batches: int | None, device: torch.device, channels_last: bool) -> Iterable[Tensor]:
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    for i, (images, _) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        yield images.to(device, non_blocking=True, memory_format=memory_format)


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    bn_modules: Iterable[nn.modules.batchnorm._BatchNorm],
    loader: DataLoader,
    num_batches: int,
    device: torch.device,
    autocast=None,
    channels_last: bool = True,
) -> None:
    """Recompute running statistics of ``bn_modules`` as a cumulative average over
    ``num_batches`` batches, with every other module in eval mode."""
    bn_modules = list(bn_modules)
    was_training = model.training
    model.eval()
    momenta = []
    for bn in bn_modules:
        bn.reset_running_stats()
        momenta.append(bn.momentum)
        bn.momentum = None  # cumulative moving average
        bn.train()
    for images in _batches(loader, num_batches, device, channels_last):
        with autocast() if autocast else torch.autocast(device.type, enabled=False):
            model(images)
    for bn, momentum in zip(bn_modules, momenta):
        bn.momentum = momentum
    model.train(was_training)


@torch.no_grad()
def bn_discrepancy(
    model: nn.Module,
    bn_modules: dict[str, nn.modules.batchnorm._BatchNorm],
    loader: DataLoader,
    num_batches: int,
    device: torch.device,
    autocast=None,
    channels_last: bool = True,
) -> dict[str, dict[str, float]]:
    """Compare each BN's running statistics with the statistics of its actual inputs
    on ``loader`` (model in eval mode).

    Returns per layer: ``mean_shift`` = max over channels of
    |data mean - running mean| / sqrt(running var + eps), and ``var_ratio_max`` /
    ``var_ratio_min`` = extreme channel ratios of data var to running var.
    """
    sums: dict[str, list[Tensor]] = {}

    def hook(name):
        def fn(module, inputs, output):
            x = inputs[0].float()
            dims = [0, 2, 3] if x.ndim == 4 else [0]
            n = x.numel() // x.shape[1]
            if name not in sums:
                zeros = torch.zeros(x.shape[1], device=x.device, dtype=torch.float64)
                sums[name] = [zeros, zeros.clone(), 0]
            s = sums[name]
            s[0] += x.sum(dims).double()
            s[1] += (x * x).sum(dims).double()
            s[2] += n

        return fn

    was_training = model.training
    model.eval()
    handles = [m.register_forward_hook(hook(name)) for name, m in bn_modules.items()]
    try:
        for images in _batches(loader, num_batches, device, channels_last):
            with autocast() if autocast else torch.autocast(device.type, enabled=False):
                model(images)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    out = {}
    for name, (s1, s2, n) in sums.items():
        bn = bn_modules[name]
        mean = s1 / n
        var = (s2 / n - mean * mean).clamp_min(0)
        run_mean, run_var = bn.running_mean.double(), bn.running_var.double()
        shift = (mean - run_mean).abs() / (run_var + bn.eps).sqrt()
        ratio = (var + bn.eps) / (run_var + bn.eps)
        out[name] = {
            "mean_shift": float(shift.max()),
            "var_ratio_max": float(ratio.max()),
            "var_ratio_min": float(ratio.min()),
        }
    return out


@torch.no_grad()
def resizer_output_stats(resizer: LearnableResizer, images: Tensor) -> dict[str, float]:
    """Size of the learned residual (output - plain resize) in eval mode, and how much
    the output changes when the resizer's BNs use batch statistics instead."""
    was_training = resizer.training
    resizer.eval()
    naive = resizer._resize(images).float()
    out_eval = resizer(images).float()
    train_bn = copy.deepcopy(resizer).train()  # batch stats without touching the original
    out_batch = train_bn(images).float()
    resizer.train(was_training)

    def rms(t: Tensor) -> float:
        return math.sqrt(float(t.pow(2).mean()))

    residual = out_eval - naive
    return {
        "naive_std": float(naive.std()),
        "out_mean": float(out_eval.mean()),
        "out_std": float(out_eval.std()),
        "residual_rms": rms(residual),
        "residual_abs_max": float(residual.abs().max()),
        "eval_vs_batch_bn_rms": rms(out_eval - out_batch),
    }
