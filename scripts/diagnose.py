"""Diagnose eval-mode val accuracy drops from per-epoch checkpoints.

For every ``epoch_XX.pt`` in a run (train with ``train.save_epoch_checkpoints=true``),
evaluates val top-1 with the stored BN running statistics and with BN statistics
recomputed on training batches (resizer only / backbone only / both), and records how
far the BN running statistics are from the statistics of the val data they normalize.

Example:
    python scripts/diagnose.py --run-dir outputs/diag_resizer368_seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from learnable_resizer.data import build_dataloaders
from learnable_resizer.diagnostics import batch_norms, bn_discrepancy, recalibrate_bn, resizer_output_stats
from learnable_resizer.engine import Precision, evaluate
from learnable_resizer.models import LearnableResizer, build_model
from learnable_resizer.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recal-batches", type=int, default=50, help="Train batches for BN recalibration.")
    parser.add_argument("--stats-batches", type=int, default=10, help="Val batches for BN discrepancy.")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt in {args.run_dir}; train with train.save_epoch_checkpoints=true")

    cfg = torch.load(checkpoints[0], map_location="cpu", weights_only=True)["config"]
    data_cfg, train_cfg = dict(cfg["data"], num_workers=args.num_workers), cfg["train"]
    channels_last = train_cfg["channels_last"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders(data_cfg, train_cfg["batch_size"])
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    precision = Precision(train_cfg["amp"], device)
    model = build_model(cfg["model"]).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    learnable = isinstance(model.resizer, LearnableResizer)
    probe_images = next(iter(val_loader))[0].to(device)

    variants = {"eval": [], "recal_backbone": ["backbone"]}
    if learnable:
        variants.update(recal_resizer=["resizer"], recal_all=["resizer", "backbone"])

    out_path = args.run_dir / "diagnostics.jsonl"
    out_path.unlink(missing_ok=True)
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        record = {"epoch": ckpt["epoch"]}

        for variant, parts in variants.items():
            model.load_state_dict(ckpt["model"])
            bns = [bn for part in parts for bn in batch_norms(getattr(model, part)).values()]
            if bns:
                set_seed(0)  # same recalibration batches for every checkpoint
                recalibrate_bn(model, bns, train_loader, args.recal_batches, device, precision.autocast, channels_last)
            stats = evaluate(model, val_loader, criterion, precision, device, channels_last)
            record[f"top1_{variant}"] = stats["top1"]
            record[f"loss_{variant}"] = stats["loss"]

        model.load_state_dict(ckpt["model"])
        tracked = {f"resizer.{k}": v for k, v in batch_norms(model.resizer).items()}
        backbone_bns = {f"backbone.{k}": v for k, v in batch_norms(model.backbone).items()}
        tracked.update(backbone_bns)
        discrepancy = bn_discrepancy(
            model, tracked, val_loader, args.stats_batches, device, precision.autocast, channels_last
        )
        record["bn"] = {k: v for k, v in discrepancy.items() if k.startswith("resizer.")}
        first_backbone_bn = next(iter(backbone_bns))
        record["bn"][first_backbone_bn] = discrepancy[first_backbone_bn]
        record["backbone_max_mean_shift"] = max(discrepancy[k]["mean_shift"] for k in backbone_bns)
        if learnable:
            with precision.autocast():
                record["resizer"] = resizer_output_stats(model.resizer, probe_images)

        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        line = " ".join(f"{v}={record[f'top1_{v}']:.2f}" for v in variants)
        line += f" | backbone max shift {record['backbone_max_mean_shift']:.2f}"
        if learnable:
            r = record["resizer"]
            line += (
                f" | residual rms {r['residual_rms']:.3f} (naive std {r['naive_std']:.3f})"
                f" eval-vs-batch-BN rms {r['eval_vs_batch_bn_rms']:.3f}"
            )
        print(f"epoch {record['epoch']}: {line}", flush=True)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
