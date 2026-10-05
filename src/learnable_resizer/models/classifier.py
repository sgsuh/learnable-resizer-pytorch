"""Resizer + backbone classifier."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from learnable_resizer.models.backbones import build_backbone
from learnable_resizer.models.resizer import BilinearResizer, LearnableResizer


class ResizerClassifier(nn.Module):
    """Applies ``resizer`` to normalized images, then classifies with ``backbone``."""

    def __init__(self, resizer: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.resizer = resizer
        self.backbone = backbone

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(self.resizer(x))


def build_resizer(resizer_cfg: dict[str, Any]) -> nn.Module:
    kind = resizer_cfg["type"]
    if kind == "bilinear":
        return BilinearResizer(
            resizer_cfg["output_size"],
            interpolation=resizer_cfg["interpolation"],
            antialias=resizer_cfg["antialias"],
        )
    if kind == "learnable":
        return LearnableResizer(
            resizer_cfg["output_size"],
            num_filters=resizer_cfg["num_filters"],
            num_res_blocks=resizer_cfg["num_res_blocks"],
            interpolation=resizer_cfg["interpolation"],
            antialias=resizer_cfg["antialias"],
            zero_init_last=resizer_cfg["zero_init_last"],
        )
    raise ValueError(f"Unknown resizer type: {kind!r}")


def build_model(model_cfg: dict[str, Any]) -> ResizerClassifier:
    resizer = build_resizer(model_cfg["resizer"])
    backbone = build_backbone(model_cfg["backbone"], model_cfg["num_classes"], model_cfg["pretrained"])
    return ResizerClassifier(resizer, backbone)
