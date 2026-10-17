"""Train and evaluation loops."""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learnable_resizer.models.resizer import resize
from learnable_resizer.utils import AverageMeter, topk_correct


class Precision:
    """Mixed-precision settings: bf16 when supported (no scaler), else fp16 + GradScaler."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled and device.type == "cuda"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=self.enabled and self.dtype == torch.float16)

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.dtype)


def _to_device(images, targets, device, channels_last):
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    images = images.to(device, non_blocking=True, memory_format=memory_format)
    return images, targets.to(device, non_blocking=True)


class ResidualTracker:
    """Forward hook on a learnable resizer: ``value`` is the mean squared residual
    (output - plain resize of the input) of the latest forward, with grad."""

    def __init__(self, resizer: nn.Module) -> None:
        self.value: torch.Tensor | None = None
        self._handle = resizer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        naive = resize(inputs[0], module.output_size, module.interpolation, module.antialias)
        self.value = (output.float() - naive.float()).pow(2).mean()

    def remove(self) -> None:
        self._handle.remove()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    precision: Precision,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_interval: int = 50,
    channels_last: bool = True,
    residual_l2: float = 0.0,
) -> dict[str, float]:
    """``loss`` in the returned stats is the classification loss; with ``residual_l2 > 0``
    the optimized loss adds ``residual_l2 * residual_sq``, and ``residual_sq`` is reported."""
    model.train()
    loss_meter, top1_meter, residual_meter = AverageMeter(), AverageMeter(), AverageMeter()
    tracker = ResidualTracker(model.resizer) if residual_l2 > 0 else None
    steps_per_epoch = len(loader)
    start = time.perf_counter()

    try:
        for step, (images, targets) in enumerate(loader):
            images, targets = _to_device(images, targets, device, channels_last)
            with precision.autocast():
                logits = model(images)
                ce_loss = criterion(logits, targets)
            loss = ce_loss
            batch_size = targets.shape[0]
            if tracker is not None:
                loss = loss + residual_l2 * tracker.value
                residual_meter.update(tracker.value.item(), batch_size)

            optimizer.zero_grad(set_to_none=True)
            precision.scaler.scale(loss).backward()
            precision.scaler.step(optimizer)
            precision.scaler.update()
            scheduler.step()

            loss_meter.update(ce_loss.item(), batch_size)
            top1_meter.update(topk_correct(logits, targets, ks=(1,))[0] / batch_size * 100, batch_size)

            global_step = epoch * steps_per_epoch + step
            if step % log_interval == 0 or step == steps_per_epoch - 1:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - start
                throughput = loss_meter.count / elapsed
                writer.add_scalar("train/loss_step", ce_loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                residual_msg = f"residual_sq {residual_meter.avg:.3f} " if tracker is not None else ""
                print(
                    f"epoch {epoch} [{step + 1}/{steps_per_epoch}] "
                    f"loss {loss_meter.avg:.4f} top1 {top1_meter.avg:.2f} {residual_msg}"
                    f"lr {lr:.5f} {throughput:.0f} img/s",
                    flush=True,
                )
    finally:
        if tracker is not None:
            tracker.remove()

    stats = {"loss": loss_meter.avg, "top1": top1_meter.avg, "time": time.perf_counter() - start}
    if tracker is not None:
        stats["residual_sq"] = residual_meter.avg
    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    precision: Precision,
    device: torch.device,
    channels_last: bool = True,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    correct1 = correct5 = total = 0

    for images, targets in loader:
        images, targets = _to_device(images, targets, device, channels_last)
        with precision.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        c1, c5 = topk_correct(logits, targets, ks=(1, 5))
        correct1, correct5, total = correct1 + c1, correct5 + c5, total + targets.shape[0]
        loss_meter.update(loss.item(), targets.shape[0])

    return {"loss": loss_meter.avg, "top1": correct1 / total * 100, "top5": correct5 / total * 100}
