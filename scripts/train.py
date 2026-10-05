"""Train a (resizer + backbone) classifier on Imagenette.

Example:
    python scripts/train.py --config configs/imagenette_resnet50_bilinear.yaml
    python scripts/train.py --config configs/imagenette_resnet50_resizer.yaml --opts train.epochs=1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from learnable_resizer.config import add_config_args, load_config, save_config
from learnable_resizer.data import build_dataloaders
from learnable_resizer.engine import Precision, evaluate, train_one_epoch
from learnable_resizer.models import LearnableResizer, build_model
from learnable_resizer.utils import (
    build_lr_scheduler,
    count_gflops,
    count_params,
    load_backbone_weights,
    param_groups,
    save_checkpoint,
    set_seed,
)
from learnable_resizer.visualize import resizer_comparison_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--resume", action="store_true", help="Resume from <run_dir>/last.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    if run_dir.exists() and not args.resume and (run_dir / "last.pt").exists():
        raise FileExistsError(f"{run_dir} already has a checkpoint; use --resume or a new run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders(data_cfg, train_cfg["batch_size"])

    model = build_model(model_cfg)
    if model_cfg["init_checkpoint"]:
        load_backbone_weights(model, model_cfg["init_checkpoint"])
        print(f"Initialized backbone from {model_cfg['init_checkpoint']}")
    model.to(device)
    if train_cfg["channels_last"]:
        model.to(memory_format=torch.channels_last)

    input_shape = (3, data_cfg["input_size"], data_cfg["input_size"])
    stats = {
        "params_resizer": count_params(model.resizer),
        "params_total": count_params(model),
        "gflops": count_gflops(model, input_shape, device),
    }
    print(f"Run: {cfg['run_name']} | {stats}")

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer = torch.optim.SGD(
        param_groups(model, train_cfg["weight_decay"]), lr=train_cfg["lr"], momentum=train_cfg["momentum"]
    )
    scheduler = build_lr_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        step_epochs=train_cfg["lr_step_epochs"],
        gamma=train_cfg["lr_gamma"],
        warmup_epochs=train_cfg["warmup_epochs"],
    )
    precision = Precision(train_cfg["amp"], device)

    start_epoch, best_top1 = 0, 0.0
    if args.resume:
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_top1 = ckpt["epoch"] + 1, ckpt["best_top1"]
        print(f"Resumed from epoch {ckpt['epoch']} (best top1 {best_top1:.2f})")

    writer = SummaryWriter(run_dir)
    writer.add_text("config", f"```yaml\n{(run_dir / 'config.yaml').read_text()}\n```")
    for key, value in stats.items():
        writer.add_scalar(f"model/{key}", value, 0)

    num_log_images = train_cfg["num_log_images"]
    log_images = None
    if isinstance(model.resizer, LearnableResizer) and num_log_images > 0:
        log_images = next(iter(val_loader))[0][:num_log_images].to(device)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            precision,
            device,
            epoch,
            writer,
            log_interval=train_cfg["log_interval"],
            channels_last=train_cfg["channels_last"],
        )
        val_stats = evaluate(model, val_loader, criterion, precision, device, train_cfg["channels_last"])

        is_best = val_stats["top1"] > best_top1
        best_top1 = max(best_top1, val_stats["top1"])
        print(
            f"epoch {epoch} | train loss {train_stats['loss']:.4f} top1 {train_stats['top1']:.2f} "
            f"| val loss {val_stats['loss']:.4f} top1 {val_stats['top1']:.2f} top5 {val_stats['top5']:.2f} "
            f"| best {best_top1:.2f} | {train_stats['time']:.0f}s",
            flush=True,
        )

        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_stats.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        if log_images is not None:
            model.eval()
            grid = resizer_comparison_grid(model.resizer, log_images, data_cfg["mean"], data_cfg["std"])
            writer.add_image("resizer/plain_learned_diff", grid, epoch)
        writer.flush()

        with open(run_dir / "metrics.jsonl", "a") as f:
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
            record.update({f"val_{k}": v for k, v in val_stats.items()})
            f.write(json.dumps(record) + "\n")

        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=epoch,
            best_top1=best_top1,
            config=cfg,
        )
        save_checkpoint(run_dir / "last.pt", **state)
        if is_best:
            save_checkpoint(run_dir / "best.pt", **state)

    final = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    summary = {
        "run_name": cfg["run_name"],
        "final_val_top1": final["val_top1"],
        "final_val_top5": final["val_top5"],
        "best_val_top1": best_top1,
        **stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    writer.close()
    print(f"Done. {summary}")


if __name__ == "__main__":
    main()
