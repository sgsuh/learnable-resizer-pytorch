"""Visual comparison of bilinear and learned resizer outputs (paper Fig. 4/6)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchvision.utils import make_grid

from learnable_resizer.models.resizer import BilinearResizer


def denormalize(x: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    """Undo per-channel normalization and clamp to [0, 1]."""
    mean_t = x.new_tensor(mean).view(1, -1, 1, 1)
    std_t = x.new_tensor(std).view(1, -1, 1, 1)
    return (x * std_t + mean_t).clamp(0.0, 1.0)


@torch.no_grad()
def resizer_comparison_grid(
    resizer: nn.Module,
    images: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    diff_gain: float = 4.0,
) -> Tensor:
    """Grid with rows: plain resize, learned resize, and amplified |difference|.

    Args:
        resizer: A resizer with ``output_size``, ``interpolation`` and ``antialias``.
        images: Normalized images at the resizer input resolution, (B, C, H, W).
    """
    baseline = BilinearResizer(resizer.output_size, resizer.interpolation, resizer.antialias)
    plain = denormalize(baseline(images), mean, std)
    learned = denormalize(resizer(images).float(), mean, std)
    diff = ((learned - plain).abs() * diff_gain).clamp(0.0, 1.0)
    return make_grid(torch.cat([plain, learned, diff]).cpu(), nrow=images.shape[0], padding=2)
