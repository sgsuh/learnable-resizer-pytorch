"""torchvision classification backbones with a replaced classification head."""

from __future__ import annotations

from torch import nn
from torchvision import models


def _replace_last_linear(model: nn.Module, num_classes: int) -> None:
    name, last = None, None
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name, last = module_name, module
    if last is None:
        raise ValueError(f"No nn.Linear head found in {type(model).__name__}")
    parent = model.get_submodule(name.rpartition(".")[0])
    setattr(parent, name.rpartition(".")[2], nn.Linear(last.in_features, num_classes))


def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a torchvision classifier (e.g. resnet50, densenet121, mobilenet_v2,
    efficientnet_b0) with its final linear layer replaced for ``num_classes``."""
    model = models.get_model(name, weights="DEFAULT" if pretrained else None)
    _replace_last_linear(model, num_classes)
    return model
