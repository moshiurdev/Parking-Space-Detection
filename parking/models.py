"""Classifiers for 64x64 slot patches.

Every model takes a float tensor ``(B, 3, 64, 64)`` with values in ``[0, 1]``.
Resizing and normalisation happen inside the model, so data loading, the
evaluation script and the web app treat all architectures the same way.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MobileNetV3Small(nn.Module):
    """torchvision MobileNetV3-small with a 2-way head."""

    def __init__(self, input_size: int = 96, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights, dropout=dropout)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
        self.net = net
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


class MAlexNet(nn.Module):
    """mAlexNet from Amato et al., "Deep learning for decentralized parking lot
    occupancy detection" (Expert Systems with Applications, 2017).

    Three conv layers (16/20/30 filters) and two fully connected layers
    (48 -> 2), about 55k parameters, 224x224 input. Trained from scratch.
    """

    def __init__(self, input_size: int = 224, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, stride=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.LocalResponseNorm(5),
            nn.Conv2d(16, 20, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.LocalResponseNorm(5),
            nn.Conv2d(20, 30, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, 3, input_size, input_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, 48), nn.ReLU(inplace=True), nn.Linear(48, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.classifier(self.features(x - 0.5))


ARCHS = ["mobilenet_v3_small", "malexnet"]


def build_model(arch: str, input_size: int | None = None, pretrained: bool = True,
                dropout: float | None = None) -> nn.Module:
    if arch == "mobilenet_v3_small":
        return MobileNetV3Small(input_size or 96, pretrained=pretrained,
                                dropout=0.2 if dropout is None else dropout)
    if arch == "malexnet":
        return MAlexNet(input_size or 224, dropout=0.0 if dropout is None else dropout)
    raise ValueError(f"unknown arch {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
