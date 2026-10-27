"""Evaluate a checkpoint on every test set and measure CPU latency per frame.

Test sets:
    pklot_test     held-out days of the training lots (PUCPR, UFPR04)
    pklot_unseen   lot UFPR05: new camera, new lot, never seen in training
    cnr_ext        CNRPark-EXT (Italy): new dataset, 9 cameras, all 23 days
    cnr_ext_later  CNR-EXT without its first 3 days (the days the fine-tuning
                   experiment trains on), so all models compare on equal data

Accuracy is computed with the PyTorch model. The checkpoint is also exported
to ONNX (``model.onnx`` next to it); CPU latency per frame is measured for
ONNX Runtime (the deployment path used by the web app) and for PyTorch eager,
both with 4 threads, and the two backends are checked to agree.

Writes ``metrics.json`` and ``confusion_matrices.png`` to ``--out``.

Example:
    python evaluate.py --checkpoint runs/mobilenetv3/best.pt --out results/mobilenetv3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from parking import CLASSES  # noqa: E402
from parking.data import load_patch_set  # noqa: E402
from parking.models import count_parameters  # noqa: E402
from parking.onnx_model import OnnxClassifier, export_onnx  # noqa: E402
from parking.predictor import SlotPredictor, softmax_occupied  # noqa: E402
from parking.slots import read_cnr_camera_csv, read_pklot_xml  # noqa: E402
from parking.splits import split_cnr_days  # noqa: E402
from parking.utils import load_checkpoint, pick_device  # noqa: E402
from train import predict  # noqa: E402

ROOT = Path(__file__).resolve().parent
TITLES = {"pklot_test": "PKLot held-out days", "pklot_unseen": "PKLot unseen lot (UFPR05)",
          "cnr_ext": "CNRPark-EXT (cross-dataset)"}


def scores(labels: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    return {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "f1_occupied": float(f1_score(labels, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),  # rows: true empty/occupied, cols: predicted
    }


def breakdown(labels, pred, groups) -> dict:
    return {g: scores(labels[groups == g], pred[groups == g]) for g in sorted(set(groups))}


def latency_frames() -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
    """One real frame per lot / camera type with all its slots."""
    frames = []
    pk = ROOT / "data" / "raw" / "pklot" / "PKLot" / "PKLot"
    for lot in ["PUCPR", "UFPR04", "UFPR05"]:
        jpg = sorted((pk / lot / "Sunny").glob("*/*.jpg"))[0]
        slots = read_pklot_xml(jpg.with_suffix(".xml"))
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        frames.append((f"PKLot {lot}", img, [s["polygon"] for s in slots]))
    cnr = ROOT / "data" / "raw" / "cnr"
    jpg = sorted((cnr / "FULL_IMAGE_1000x750" / "SUNNY").glob("*/camera1/*.jpg"))[0]
    img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
    frames.append(("CNR-EXT camera1", img, list(read_cnr_camera_csv(cnr / "camera1.csv").values())))
    return frames


def measure_latency(model_path: Path, repeats: int = 30) -> dict:
    predictor = SlotPredictor(model_path, device="cpu", threads=4)
    out = {"backend": predictor.backend, "threads": 4}
    for name, img, polys in latency_frames():
        for _ in range(5):
            predictor.predict(img, polys)
        runs = [predictor.predict(img, polys)[1] for _ in range(repeats)]
        out[name] = {"slots": len(polys), "image": f"{img.shape[1]}x{img.shape[0]}",
                     **{k: float(np.median([r[k] for r in runs])) for k in runs[0]}}
    return out


def check_onnx(onnx_path: Path, model: torch.nn.Module, device, images: np.ndarray) -> float:
    """Largest difference in P(occupied) between ONNX Runtime and PyTorch."""
    ref = predict(model, images, device)
    x = images.transpose(0, 3, 1, 2).astype(np.float32) / 255
    got = softmax_occupied(OnnxClassifier(onnx_path)(x))
    return float(np.abs(got - ref).max())


def plot_confusions(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, TITLES):
        cm = np.array(results[name]["overall"]["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.1%}\n({cm[i, j]:,})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=10)
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc = results[name]["overall"]["accuracy"]
        ax.set_title(f"{TITLES[name]}\naccuracy {acc:.1%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-latency", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    results = {"checkpoint": str(args.checkpoint), "name": cfg["name"], "arch": cfg["arch"],
               "input_size": cfg["input_size"], "params": count_parameters(model), "epoch": ckpt["epoch"],
               "augment": cfg["augment"], "train_set": cfg["train_set"], "train_filter": cfg["train_filter"],
               "init": cfg.get("init")}
    sets = {}
    for name in ["pklot_test", "pklot_unseen", "cnr_ext"]:
        ps = load_patch_set(name)
        prob = predict(model, ps.images, device)
        pred = (prob >= 0.5).astype(int)
        sets[name] = {"overall": scores(ps.labels, pred),
                      "by_weather": breakdown(ps.labels, pred, ps.column("weather"))}
        if name != "pklot_unseen":
            sets[name]["by_camera"] = breakdown(ps.labels, pred, ps.column("camera"))
        if name == "cnr_ext":
            days = split_cnr_days(list(ps.column("date")))
            later = np.array([days[d] == "later" for d in ps.column("date")])
            sets["cnr_ext_later"] = {"overall": scores(ps.labels[later], pred[later]),
                                     "by_weather": breakdown(ps.labels[later], pred[later],
                                                             ps.column("weather")[later])}
        print(f"{name}: acc {sets[name]['overall']['accuracy']:.4f} "
              f"macro-F1 {sets[name]['overall']['macro_f1']:.4f}", flush=True)
    results["sets"] = sets
    onnx_path = export_onnx(model, cfg, args.checkpoint.parent / "model.onnx")
    model.to(device)
    sample = np.asarray(load_patch_set("cnr_ext").images[:2000])
    results["onnx_max_abs_diff"] = check_onnx(onnx_path, model, device, sample)
    print(f"ONNX vs PyTorch max |dP|: {results['onnx_max_abs_diff']:.2e}")
    if not args.no_latency:
        results["cpu_latency_ms"] = {"onnxruntime": measure_latency(onnx_path),
                                     "pytorch": measure_latency(args.checkpoint)}
        print(json.dumps(results["cpu_latency_ms"], indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_confusions(sets, args.out / "confusion_matrices.png")
    for f in ["config.json", "history.json"]:
        src = args.checkpoint.parent / f
        if src.exists():
            (args.out / f).write_text(src.read_text())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
