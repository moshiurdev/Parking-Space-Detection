"""Batched data augmentation that runs on the training device.

Patches are small (64x64) and already in memory, so augmenting whole batches
with tensor ops on the GPU is much faster than per-sample CPU transforms.
Each sample still gets its own random parameters.

Two presets:

* ``standard``: flips, mild scale/shift/rotation, mild colour jitter.
* ``strong``: aimed at the cross-site gap. Wider geometry (loose crops, any
  90-degree orientation, aspect changes), strong colour/greyscale/blur/noise,
  random shadows and occluding blocks (trees, lamp posts and other cars cause
  most CNRPark-EXT errors).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PRESETS = ["none", "standard", "strong"]


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


def _geometric(x: torch.Tensor, strong: bool) -> torch.Tensor:
    n, dev = x.shape[0], x.device
    angle = _uniform(n, -15, 15, dev) if strong else _uniform(n, -5, 5, dev)
    if strong:  # slots appear in any orientation across cameras
        angle = angle + 90 * torch.randint(0, 4, (n,), device=dev)
    angle = angle * math.pi / 180
    scale = _uniform(n, 0.75, 1.15, dev) if strong else _uniform(n, 0.9, 1.05, dev)
    aspect = torch.exp(_uniform(n, -0.25, 0.25, dev)) if strong else torch.ones(n, device=dev)
    sx, sy = scale * aspect.sqrt(), scale / aspect.sqrt()
    shift = 0.12 if strong else 0.06
    tx, ty = _uniform(n, -shift, shift, dev), _uniform(n, -shift, shift, dev)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle), torch.sin(angle)
    # theta maps output coords to input coords; scale < 1 zooms in
    theta = torch.stack([
        torch.stack([cos * sx * flip, -sin * sy, tx], 1),
        torch.stack([sin * sx * flip, cos * sy, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _photometric(x: torch.Tensor, strong: bool) -> torch.Tensor:
    n, dev = x.shape[0], x.device
    s = 0.4 if strong else 0.2
    b = _uniform(n, 1 - s, 1 + s, dev).view(n, 1, 1, 1)
    c = _uniform(n, 1 - s, 1 + s, dev).view(n, 1, 1, 1)
    sat = _uniform(n, 1 - s, 1 + s, dev).view(n, 1, 1, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    x = gray + (x - gray) * sat
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    x = x * b
    if strong:
        # per-channel gain (white balance / colour cast)
        x = x * _uniform(3 * n, 0.9, 1.1, dev).view(n, 3, 1, 1)
        # random greyscale (p=0.2)
        g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).expand_as(x)
        x = torch.where((torch.rand(n, 1, 1, 1, device=dev) < 0.2), g, x)
    return x


def _shadow(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Darken the part of the patch on one side of a random line."""
    n, _, h, w = x.shape
    dev = x.device
    ys = torch.linspace(-1, 1, h, device=dev).view(1, h, 1)
    xs = torch.linspace(-1, 1, w, device=dev).view(1, 1, w)
    a = _uniform(n, 0, 2 * math.pi, dev).view(n, 1, 1)
    off = _uniform(n, -0.6, 0.6, dev).view(n, 1, 1)
    mask = (xs * torch.cos(a) + ys * torch.sin(a) > off).float().unsqueeze(1)
    dark = _uniform(n, 0.4, 0.8, dev).view(n, 1, 1, 1)
    apply = (torch.rand(n, 1, 1, 1, device=dev) < p).float()
    return x * (1 - apply * mask * (1 - dark))


def _blur(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device)
    k = (k[:, None] * k[None, :]) / 256
    k = k.expand(3, 1, 5, 5)
    blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k, groups=3)
    apply = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p
    return torch.where(apply, blurred, x)


def _occlude(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Random erasing with a block of random flat colour (max 1/4 of the area)."""
    n, _, h, w = x.shape
    dev = x.device
    bh = (_uniform(n, 0.15, 0.5, dev) * h).long()
    bw = (_uniform(n, 0.15, 0.5, dev) * w).long()
    y0 = (torch.rand(n, device=dev) * (h - bh)).long()
    x0 = (torch.rand(n, device=dev) * (w - bw)).long()
    ys = torch.arange(h, device=dev).view(1, h, 1)
    xs = torch.arange(w, device=dev).view(1, 1, w)
    mask = ((ys >= y0.view(n, 1, 1)) & (ys < (y0 + bh).view(n, 1, 1))
            & (xs >= x0.view(n, 1, 1)) & (xs < (x0 + bw).view(n, 1, 1))).unsqueeze(1)
    mask &= torch.rand(n, 1, 1, 1, device=dev) < p
    colour = torch.rand(n, 3, 1, 1, device=dev)
    return torch.where(mask, colour, x)


def augment(x: torch.Tensor, preset: str = "standard") -> torch.Tensor:
    """Augment a float batch ``(B, 3, H, W)`` in ``[0, 1]``."""
    if preset == "none":
        return x
    if preset not in PRESETS:
        raise ValueError(f"unknown augmentation preset {preset!r}")
    strong = preset == "strong"
    x = _geometric(x, strong)
    x = _photometric(x, strong)
    if strong:
        x = _shadow(x)
        x = _blur(x)
        x = x + torch.randn_like(x) * _uniform(x.shape[0], 0, 0.04, x.device).view(-1, 1, 1, 1)
        x = _occlude(x)
    return x.clamp_(0, 1)
