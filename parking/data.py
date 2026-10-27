"""Patch sets built by ``scripts/prepare_data.py``.

Each set is two files in ``data/patches/``:

* ``<name>.npy``: uint8 array ``(N, 64, 64, 3)``, RGB
* ``<name>.csv``: one row per patch with ``label`` (0 empty, 1 occupied) and
  metadata (``dataset, lot/camera, weather, date, image, slot``) used to break
  results down by condition.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "patches"


@dataclass
class PatchSet:
    name: str
    images: np.ndarray        # (N, H, W, 3) uint8, possibly memory-mapped
    labels: np.ndarray        # (N,) int64
    meta: list[dict]

    def __len__(self) -> int:
        return len(self.labels)

    def column(self, key: str) -> np.ndarray:
        return np.array([m[key] for m in self.meta])

    def subset(self, mask: np.ndarray, name: str | None = None) -> "PatchSet":
        idx = np.flatnonzero(mask)
        return PatchSet(name or self.name, self.images[idx], self.labels[idx], [self.meta[i] for i in idx])


def load_patch_set(name: str, patch_dir: Path = PATCH_DIR, in_memory: bool = False) -> PatchSet:
    images = np.load(patch_dir / f"{name}.npy", mmap_mode=None if in_memory else "r")
    with open(patch_dir / f"{name}.csv", newline="") as f:
        meta = list(csv.DictReader(f))
    labels = np.array([int(m["label"]) for m in meta], dtype=np.int64)
    assert len(labels) == len(images), f"{name}: {len(labels)} labels vs {len(images)} images"
    return PatchSet(name, images, labels, meta)


class PatchWriter:
    """Write a patch set chunk by chunk without holding it all in memory."""

    def __init__(self, name: str, patch_dir: Path = PATCH_DIR):
        patch_dir.mkdir(parents=True, exist_ok=True)
        self.npy = patch_dir / f"{name}.npy"
        self.csv = patch_dir / f"{name}.csv"
        self.tmp = patch_dir / f"{name}.raw.tmp"
        self.f = open(self.tmp, "wb")
        self.meta: list[dict] = []
        self.shape: tuple | None = None

    def add(self, images: np.ndarray, meta: list[dict]) -> None:
        assert len(images) == len(meta)
        self.shape = self.shape or images.shape[1:]
        self.f.write(np.ascontiguousarray(images, dtype=np.uint8).tobytes())
        self.meta += meta

    def close(self) -> None:
        self.f.close()
        n = len(self.meta)
        raw = np.memmap(self.tmp, dtype=np.uint8, mode="r", shape=(n, *self.shape))
        out = np.lib.format.open_memmap(self.npy, mode="w+", dtype=np.uint8, shape=(n, *self.shape))
        for i in range(0, n, 20000):
            out[i:i + 20000] = raw[i:i + 20000]
        out.flush()
        del out, raw
        self.tmp.unlink()
        with open(self.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.meta[0]))
            writer.writeheader()
            writer.writerows(self.meta)


def to_tensor(batch: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """uint8 (B, H, W, 3) -> float (B, 3, H, W) in [0, 1] on ``device``."""
    x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
    return x.permute(0, 3, 1, 2).float().div_(255)


def class_weights(labels: np.ndarray, mode: str = "inverse") -> torch.Tensor:
    """Per-class loss weights, normalised to mean 1."""
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if mode == "none":
        w = np.ones(2)
    elif mode == "inverse":
        w = counts.sum() / np.maximum(counts, 1)
    elif mode == "sqrt":
        w = np.sqrt(counts.sum() / np.maximum(counts, 1))
    else:
        raise ValueError(mode)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def iterate_batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        # sorted indices make memory-mapped reads sequential
        yield np.sort(order[i:i + batch_size])
