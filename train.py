"""Train a slot occupancy classifier on 64x64 patches.

Model selection uses the validation days of the training lots (``pklot_val``).
The test sets (held-out PKLot days, the unseen lot UFPR05 and CNR-EXT) are
only touched by ``evaluate.py``.

Examples:
    python train.py --arch mobilenet_v3_small --name mobilenetv3
    python train.py --arch malexnet --no-pretrained --lr 3e-3 --name malexnet
    python train.py --arch mobilenet_v3_small --augment strong --name mobilenetv3_strong
    # weather shift: train and select on sunny frames only
    python train.py --train-filter weather=sunny --val-filter weather=sunny --name mobilenetv3_sunny
    # brief fine-tuning on the first CNR-EXT days (no model selection: last epoch is kept)
    python train.py --init runs/mobilenetv3/best.pt --train-set cnr_ext \\
        --train-filter date=2015-11-12,2015-11-16,2015-11-20 --val-set none \\
        --epochs 2 --lr 3e-4 --warmup-epochs 0 --name mobilenetv3_cnr_ft
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from parking.augment import PRESETS, augment
from parking.data import PatchSet, class_weights, iterate_batches, load_patch_set, to_tensor
from parking.models import ARCHS, build_model, count_parameters
from parking.utils import pick_device, save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a parking slot occupancy classifier")
    p.add_argument("--arch", default="mobilenet_v3_small", choices=ARCHS)
    p.add_argument("--input-size", type=int, default=None,
                   help="model input resolution (default: 96 for MobileNetV3, 224 for mAlexNet)")
    p.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    p.add_argument("--init", default=None, help="start from this checkpoint (fine-tuning)")
    p.add_argument("--train-set", default="pklot_train")
    p.add_argument("--val-set", default="pklot_val", help="'none' disables model selection")
    p.add_argument("--train-filter", default=None, help="key=v1,v2 on the patch metadata")
    p.add_argument("--val-filter", default=None)
    p.add_argument("--augment", default="standard", choices=PRESETS)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--samples-per-epoch", type=int, default=0, help="random patches per epoch (0 = all)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3, help="peak AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--class-weighting", default="inverse", choices=["none", "sqrt", "inverse"])
    p.add_argument("--val-max", type=int, default=40000, help="fixed random subset of the val set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="use only N train/val samples (smoke tests)")
    p.add_argument("--name", default=None, help="run directory name under runs/")
    return p.parse_args()


def apply_filter(ps: PatchSet, spec: str | None) -> PatchSet:
    if not spec:
        return ps
    key, values = spec.split("=", 1)
    return ps.subset(np.isin(ps.column(key), values.split(",")), f"{ps.name}[{spec}]")


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """Softmax probability of 'occupied' for every patch."""
    model.eval()
    out = []
    for idx in iterate_batches(len(images), batch_size, shuffle=False):
        logits = model(to_tensor(images[idx], device))
        out.append(logits.softmax(1)[:, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    name = args.name or f"{args.arch}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = apply_filter(load_patch_set(args.train_set), args.train_filter)
    if args.limit:
        train = train.subset(np.isin(np.arange(len(train)), rng.choice(len(train), args.limit, replace=False)))
    train_images = np.asarray(train.images)  # load into memory
    val = None
    if args.val_set != "none":
        val = apply_filter(load_patch_set(args.val_set), args.val_filter)
        n_val = min(len(val), args.limit or args.val_max)
        keep = np.zeros(len(val), bool)
        keep[np.random.default_rng(0).choice(len(val), n_val, replace=False)] = True
        val = val.subset(keep)
        val_images = np.asarray(val.images)

    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        args.arch, args.input_size, args.dropout = cfg["arch"], cfg["input_size"], cfg["dropout"]
        model = build_model(args.arch, input_size=args.input_size, pretrained=False, dropout=args.dropout)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = build_model(args.arch, input_size=args.input_size, pretrained=not args.no_pretrained,
                            dropout=args.dropout)
        args.input_size = model.input_size
        args.dropout = args.dropout if args.dropout is not None else (0.2 if args.arch != "malexnet" else 0.0)
    model.to(device)

    weights = class_weights(train.labels, args.class_weighting).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_epoch = min(args.samples_per_epoch or len(train), len(train))
    steps_per_epoch = math.ceil(per_epoch / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    config = {**vars(args), "name": name, "device": str(device), "params": count_parameters(model),
              "train_size": len(train), "train_occupied": int(train.labels.sum()),
              "val_size": len(val) if val else 0, "class_weights": weights.tolist()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    history, best, step = [], None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(train))[:per_epoch]
        run_loss = run_correct = seen = 0
        for i in range(0, len(order), args.batch_size):
            idx = np.sort(order[i:i + args.batch_size])
            x = augment(to_tensor(train_images[idx], device), args.augment)
            y = torch.from_numpy(train.labels[idx]).to(device)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, args.lr)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(idx)

        record = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": run_correct / seen,
                  "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        if val is not None:
            prob = predict(model, val_images, device)
            pred = (prob >= 0.5).astype(int)
            eps = 1e-7
            p1 = np.clip(prob, eps, 1 - eps)
            record["val_loss"] = float(-np.mean(val.labels * np.log(p1) + (1 - val.labels) * np.log(1 - p1)))
            record["val_acc"] = float((pred == val.labels).mean())
            record["val_macro_f1"] = float(f1_score(val.labels, pred, average="macro"))
        history.append(record)
        print(json.dumps(record), flush=True)

        extra = {"epoch": epoch, "history": history}
        save_checkpoint(run_dir / "last.pt", model, config, extra)
        # selection: best val macro-F1, ties broken by lower val loss
        key = (record["val_macro_f1"], -record["val_loss"]) if val is not None else (epoch,)
        if best is None or key > best:
            best = key
            save_checkpoint(run_dir / "best.pt", model, config, extra)
            print(f"  new best (epoch {epoch})", flush=True)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done; best checkpoint at {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
