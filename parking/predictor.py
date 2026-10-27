"""Frame-level inference: crop every slot of a lot image and classify it.

Used by the web app and by the CPU latency benchmark in ``evaluate.py``.
A ``.onnx`` file runs in ONNX Runtime (the deployment path); a ``.pt``
checkpoint runs in PyTorch.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .data import to_tensor
from .onnx_model import OnnxClassifier
from .slots import crop_slots
from .utils import load_checkpoint


def softmax_occupied(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(1)).astype(np.float32)


class SlotPredictor:
    def __init__(self, model_path: Path, device: str = "cpu", threshold: float = 0.5, threads: int = 4):
        model_path = Path(model_path)
        self.threshold = threshold
        if model_path.suffix == ".onnx":
            self.backend = "onnxruntime"
            self.onnx = OnnxClassifier(model_path, threads=threads)
            self.config = self.onnx.config
        else:
            self.backend = "pytorch"
            self.device = torch.device(device)
            if self.device.type == "cpu":
                torch.set_num_threads(threads)
            self.model, ckpt = load_checkpoint(model_path, self.device)
            self.config = ckpt["config"]

    def _classify(self, patches: np.ndarray) -> np.ndarray:
        if self.backend == "onnxruntime":
            x = patches.transpose(0, 3, 1, 2).astype(np.float32) / 255
            return softmax_occupied(self.onnx(x))
        with torch.inference_mode():
            return self.model(to_tensor(patches, self.device)).softmax(1)[:, 1].float().cpu().numpy()

    def predict(self, image_rgb: np.ndarray, polygons: list[np.ndarray]) -> tuple[np.ndarray, dict]:
        """Probability that each slot is occupied, plus timings in ms."""
        t0 = time.perf_counter()
        patches = crop_slots(image_rgb, polygons)
        t1 = time.perf_counter()
        prob = self._classify(patches) if len(patches) else np.zeros(0, dtype=np.float32)
        t2 = time.perf_counter()
        return prob, {"crop_ms": (t1 - t0) * 1000, "model_ms": (t2 - t1) * 1000, "total_ms": (t2 - t0) * 1000}
