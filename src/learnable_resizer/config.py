"""YAML config loading with command-line overrides.

Usage:
    cfg = load_config("configs/foo.yaml", overrides=["train.lr=0.01", "model.resizer.type=learnable"])

A YAML file only needs to specify values that differ from ``DEFAULTS``. Unknown keys
raise an error to catch typos.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": None,  # defaults to the config file stem
    "output_dir": "outputs",
    "seed": 0,
    "data": {
        "root": "data/imagenette2",
        "input_size": 224,  # resizer input resolution produced by the data pipeline
        "train_crop_scale": [0.35, 1.0],
        "val_crop_ratio": 0.875,  # val: resize to input_size / ratio, then center crop
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_workers": 16,
    },
    "model": {
        "backbone": "resnet50",
        "pretrained": True,  # torchvision ImageNet weights
        "num_classes": 10,
        "init_checkpoint": None,  # load backbone weights from a previous run's checkpoint
        "resizer": {
            "type": "bilinear",  # "bilinear" | "learnable"
            "output_size": 224,
            "num_filters": 16,
            "num_res_blocks": 1,
            "interpolation": "bilinear",
            "antialias": False,
            "zero_init_last": False,
        },
    },
    "train": {
        "epochs": 10,
        "batch_size": 64,
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 1.0e-4,  # not applied to biases and normalization parameters
        "label_smoothing": 0.1,
        "lr_step_epochs": 2,
        "lr_gamma": 0.94,
        "warmup_epochs": 0,  # linear LR warmup (not in the paper; useful from scratch)
        "amp": True,
        "channels_last": True,
        "log_interval": 50,  # steps
        "num_log_images": 8,  # resizer output samples logged to TensorBoard per epoch
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any], prefix: str = "") -> None:
    for key, value in update.items():
        path = f"{prefix}{key}"
        if key not in base:
            raise KeyError(f"Unknown config key: {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Config key {path} must be a mapping, got {value!r}")
            _merge(base[key], value, prefix=f"{path}.")
        else:
            base[key] = value


def _parse_override(override: str) -> dict[str, Any]:
    if "=" not in override:
        raise ValueError(f"Override must look like key.subkey=value, got {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    nested: dict[str, Any] = {}
    node = nested
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw_value)
    return nested


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
        if cfg["run_name"] is None:
            cfg["run_name"] = Path(path).stem
    for override in overrides or []:
        _merge(cfg, _parse_override(override))
    if cfg["run_name"] is None:
        cfg["run_name"] = "default"
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides, e.g. --opts train.lr=0.01 model.resizer.type=learnable",
    )


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
