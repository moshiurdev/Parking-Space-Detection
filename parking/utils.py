"""Small shared helpers: device selection, seeding, checkpoint I/O."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, **(extra or {})}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild the model described by the checkpoint's config and load weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = build_model(cfg["arch"], input_size=cfg["input_size"], pretrained=False,
                        dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
