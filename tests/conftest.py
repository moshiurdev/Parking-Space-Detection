import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parking.models import build_model, count_parameters  # noqa: E402
from parking.utils import save_checkpoint  # noqa: E402


def lot_image(width: int = 320, height: int = 200) -> tuple[np.ndarray, list[np.ndarray]]:
    """Synthetic grey lot with two slots: left one empty, right one with a red 'car'."""
    img = np.full((height, width, 3), 90, np.uint8)
    slots = [np.float32([[20, 40], [120, 40], [120, 160], [20, 160]]),
             np.float32([[180, 40], [280, 40], [280, 160], [180, 160]])]
    cv2.rectangle(img, (195, 55), (265, 145), (200, 30, 30), -1)
    return img, slots


def marked_lot_image() -> tuple[np.ndarray, list[np.ndarray]]:
    """Synthetic top-down lot (BGR, 1000x800) with painted stall lines.

    Two bays of 6 stalls (100 px wide, 250 px long) side by side, split by a
    centre line, with a few dark and white cars and a planter above the left
    bay. Returns the image and the 12 stalls between the painted lines."""
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(105, 6, (800, 1000, 3)), 0, 255).astype(np.uint8)
    white = (235, 235, 235)
    for y in range(100, 701, 100):
        cv2.line(img, (100, y), (600, y), white, 6)
    cv2.line(img, (350, 100), (350, 700), white, 6)
    cv2.rectangle(img, (110, 15), (340, 85), (40, 140, 40), -1)  # planter (BGR green)
    for x, y, color in [(130, 115, (40, 40, 45)), (380, 315, (225, 225, 230)), (130, 515, (30, 30, 160))]:
        cv2.rectangle(img, (x, y), (x + 190, y + 70), color, -1)
        cv2.rectangle(img, (x + 120, y + 10), (x + 160, y + 60), (50, 50, 50), -1)  # windscreen
    stalls = [np.float32([[x0, y], [x0 + 250, y], [x0 + 250, y + 100], [x0, y + 100]])
              for x0 in (100, 350) for y in range(100, 700, 100)]
    return img, stalls


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """Randomly initialised mAlexNet: tests run without the dataset or trained weights."""
    model = build_model("malexnet")
    cfg = {"name": "test", "arch": "malexnet", "input_size": 224, "dropout": 0.0,
           "params": count_parameters(model)}
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, model, cfg, {"epoch": 0})
    return path


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("demo")
    img, slots = lot_image()
    for i in range(3):
        cv2.imwrite(str(d / f"frame{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    seq = {"id": "test", "title": "Test lot", "description": "synthetic",
           "frames": [{"image": str(d / f"frame{i}.jpg"), "time": f"t{i}", "weather": "sunny",
                       "slots": [{"id": j, "polygon": s.tolist(), "label": j} for j, s in enumerate(slots)]}
                      for i in range(3)]}
    (d / "sequences.json").write_text(json.dumps({"sequences": [seq]}))
    return d
