"""Imagenette datasets and transforms.

The data pipeline produces images at the *resizer input* resolution
(``data.input_size``). The model's resizer then maps them to the backbone resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T

# WNID -> human-readable class name.
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def build_transforms(data_cfg: dict[str, Any], train: bool) -> T.Compose:
    size = data_cfg["input_size"]
    if train:
        spatial = [
            T.RandomResizedCrop(size, scale=tuple(data_cfg["train_crop_scale"]), antialias=True),
            T.RandomHorizontalFlip(),
        ]
    else:
        spatial = [
            T.Resize(round(size / data_cfg["val_crop_ratio"]), antialias=True),
            T.CenterCrop(size),
        ]
    return T.Compose(
        [
            T.RGB(),  # a few Imagenette images are grayscale
            *spatial,
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ]
    )


def build_dataloaders(data_cfg: dict[str, Any], batch_size: int) -> tuple[DataLoader, DataLoader]:
    root = Path(data_cfg["root"])
    train_set = datasets.ImageFolder(root / "train", transform=build_transforms(data_cfg, train=True))
    val_set = datasets.ImageFolder(root / "val", transform=build_transforms(data_cfg, train=False))
    if train_set.classes != val_set.classes:
        raise RuntimeError("train and val class folders differ")

    common = dict(
        batch_size=batch_size,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def build_recal_loader(train_loader: DataLoader, generator: torch.Generator) -> DataLoader:
    """Loader over the train set for BN recalibration. It shuffles with its own
    ``generator`` (so the training order and global RNG are unaffected) and does not keep
    workers alive, since it is abandoned after a few batches."""
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_loader.num_workers,
        pin_memory=True,
        generator=generator,
    )
