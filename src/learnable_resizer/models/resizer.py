"""Learnable image resizer (Talebi & Milanfar, ICCV 2021, Fig. 3)."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

_ALIGN_CORNERS_MODES = {"linear", "bilinear", "bicubic", "trilinear"}
_ANTIALIAS_MODES = {"bilinear", "bicubic"}


def _to_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"output_size must be an int or (height, width), got {size!r}")
    return (int(size[0]), int(size[1]))


def resize(
    x: Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    antialias: bool = False,
) -> Tensor:
    """Resize a (B, C, H, W) tensor with TF2-compatible half-pixel sampling.

    ``align_corners=False`` matches TF2's ``tf.image.resize`` / Keras ``Resizing``.
    """
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if antialias and mode not in _ANTIALIAS_MODES:
        raise ValueError(f"antialias is only supported for {sorted(_ANTIALIAS_MODES)}, got {mode!r}")
    align_corners = False if mode in _ALIGN_CORNERS_MODES else None
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners, antialias=antialias)


class BilinearResizer(nn.Module):
    """Parameter-free baseline resizer sharing the interface of ``LearnableResizer``."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        interpolation: str = "bilinear",
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.output_size = _to_size(output_size)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, x: Tensor) -> Tensor:
        return resize(x, self.output_size, self.interpolation, self.antialias)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, interpolation={self.interpolation!r}, antialias={self.antialias}"


class ResBlock(nn.Module):
    """Conv3x3 -> BN -> LeakyReLU -> Conv3x3 -> BN, plus identity skip."""

    def __init__(self, num_filters: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.act = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class LearnableResizer(nn.Module):
    """CNN resizer jointly trained with a downstream vision model.

    The output is ``resize(x) + residual(x)``, where the residual is computed from
    features at the input resolution, resized to ``output_size`` (feature bottleneck),
    and refined by ``num_res_blocks`` residual blocks. Works for any input size and
    both down- and up-scaling.

    Args:
        output_size: Target (height, width), or a single int for square outputs.
        in_channels: Number of image channels.
        num_filters: Width ``n`` of the intermediate convolutions.
        num_res_blocks: Number ``r`` of residual blocks.
        interpolation: ``F.interpolate`` mode for both the image and feature resizes.
        antialias: Use anti-aliasing when down-scaling (bilinear/bicubic only).
        negative_slope: LeakyReLU negative slope.
        zero_init_last: Zero-initialize the last conv so the module starts out as the
            plain ``interpolation`` resizer.
    """

    def __init__(
        self,
        output_size: int | Sequence[int],
        in_channels: int = 3,
        num_filters: int = 16,
        num_res_blocks: int = 1,
        interpolation: str = "bilinear",
        antialias: bool = False,
        negative_slope: float = 0.2,
        zero_init_last: bool = False,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")
        self.output_size = _to_size(output_size)
        self.interpolation = interpolation
        self.antialias = antialias

        # Features at the input resolution.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 7, padding=3),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(num_filters, num_filters, 1),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(num_filters),
        )
        # Features at the output resolution.
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters, negative_slope) for _ in range(num_res_blocks)]
        )
        self.projection = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )
        self.tail = nn.Conv2d(num_filters, in_channels, 7, padding=3)

        if zero_init_last:
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)

    def _resize(self, x: Tensor) -> Tensor:
        return resize(x, self.output_size, self.interpolation, self.antialias)

    def forward(self, x: Tensor) -> Tensor:
        naive = self._resize(x)
        bottleneck = self._resize(self.head(x))
        features = bottleneck + self.projection(self.res_blocks(bottleneck))
        return naive + self.tail(features)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, interpolation={self.interpolation!r}, antialias={self.antialias}"


def count_conv_weights(module: nn.Module) -> int:
    """Number of conv kernel weights (no bias / BN), as counted in the paper's Table 2."""
    return sum(m.weight.numel() for m in module.modules() if isinstance(m, nn.Conv2d))

