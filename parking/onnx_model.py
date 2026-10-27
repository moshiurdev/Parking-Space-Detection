"""ONNX export and ONNX Runtime inference for CPU deployment.

PyTorch's eager CPU path is slow for MobileNetV3's depthwise convolutions on
the macOS arm64 build used here (no oneDNN), so the demo and the latency
benchmark run the exported model in ONNX Runtime. The exported graph includes
the model's own resize and normalisation, so it takes the same input as the
PyTorch model: float32 ``(N, 3, 64, 64)`` in ``[0, 1]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import PATCH_SIZE


def export_onnx(model: torch.nn.Module, config: dict, path: Path) -> Path:
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.rand(2, 3, PATCH_SIZE, PATCH_SIZE)
    torch.onnx.export(model, (dummy,), str(path), input_names=["patches"], output_names=["logits"],
                      dynamic_axes={"patches": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    proto = onnx.load(str(path))
    entry = proto.metadata_props.add()
    entry.key, entry.value = "config", json.dumps(config)
    onnx.save(proto, str(path))
    return path


class OnnxClassifier:
    """Callable like the PyTorch model, but on numpy arrays: returns logits."""

    def __init__(self, path: Path, threads: int = 4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.config = json.loads(meta["config"]) if "config" in meta else {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"patches": np.ascontiguousarray(x, dtype=np.float32)})[0]
