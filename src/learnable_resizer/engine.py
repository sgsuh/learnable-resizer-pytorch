"""Train and evaluation loops."""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
) -> dict[str, float]:
    model.train()
    loss_meter, top1_meter = AverageMeter(), AverageMeter()
    steps_per_epoch = len(loader)
    start = time.perf_counter()

    for step, (images, targets) in enumerate(loader):
        images, targets = _to_device(images, targets, device, channels_last)
        with precision.autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        precision.scaler.scale(loss).backward()
        precision.scaler.step(optimizer)
        precision.scaler.update()
        scheduler.step()

        batch_size = targets.shape[0]
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(topk_correct(logits, targets, ks=(1,))[0] / batch_size * 100, batch_size)

        global_step = epoch * steps_per_epoch + step
        if step % log_interval == 0 or step == steps_per_epoch - 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - start
            throughput = loss_meter.count / elapsed
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            print(
                f"epoch {epoch} [{step + 1}/{steps_per_epoch}] "
                f"loss {loss_meter.avg:.4f} top1 {top1_meter.avg:.2f} "
                f"lr {lr:.5f} {throughput:.0f} img/s",
                flush=True,
            )

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "time": time.perf_counter() - start}


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
