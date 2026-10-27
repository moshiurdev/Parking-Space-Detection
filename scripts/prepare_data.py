"""Cut slot patches out of PKLot and CNR-EXT and write the patch sets.

PKLot (full 1280x720 frames + XML polygons):

* ``pklot_train`` / ``pklot_val`` / ``pklot_test``: lots PUCPR and UFPR04,
  split by whole days (see ``parking/splits.py``). Training days keep every
  second frame (frames are 5 minutes apart and nearly identical); validation
  and test days keep every frame.
* ``pklot_unseen``: every frame of lot UFPR05, a camera and lot that is never
  used for training or model selection.

CNR-EXT (official 150x150 patches, labels from ``CNRPark+EXT.csv``):

* ``cnr_ext``: all 144,965 patches, 9 cameras, 23 days.

The day assignment is written to ``results/splits.json``.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parking import PATCH_SIZE  # noqa: E402
from parking.data import PatchWriter  # noqa: E402
from parking.slots import crop_slots, read_pklot_xml  # noqa: E402
from parking.splits import SOURCE_LOTS, UNSEEN_LOT, split_cnr_days, split_days  # noqa: E402

PKLOT_DIR = ROOT / "data" / "raw" / "pklot" / "PKLot" / "PKLot"
CNR_DIR = ROOT / "data" / "raw" / "cnr"


def pklot_images() -> list[dict]:
    items = []
    for jpg in sorted(PKLOT_DIR.glob("*/*/*/*.jpg")):
        lot, weather, date = jpg.parts[-4:-1]
        items.append({"lot": lot, "weather": weather.lower(), "date": date, "path": jpg})
    return items


def _cut_pklot(item: dict):
    xml = item["path"].with_suffix(".xml")
    if not xml.exists():
        return item, None, "no xml"
    try:
        slots = [s for s in read_pklot_xml(xml) if s["occupied"] is not None]
    except Exception as e:  # a handful of PKLot XML files are truncated
        return item, None, f"bad xml: {e}"
    img = cv2.imread(str(item["path"]))
    if img is None:
        return item, None, "unreadable image"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches = crop_slots(rgb, [s["polygon"] for s in slots])
    return item, (patches, [(s["id"], s["occupied"]) for s in slots]), None


def build_pklot_sets(items: list[dict], workers: int) -> dict[str, list[dict]]:
    by_set = defaultdict(list)
    for lot in sorted({i["lot"] for i in items}):
        lot_items = [i for i in items if i["lot"] == lot]
        if lot == UNSEEN_LOT:
            by_set["pklot_unseen"] += lot_items
            continue
        assert lot in SOURCE_LOTS, lot
        day_weather = defaultdict(list)
        for i in lot_items:
            day_weather[i["date"]].append(i["weather"])
        days = split_days(day_weather)
        frame_no = Counter()
        for i in lot_items:
            split = days[i["date"]]
            frame_no[(lot, i["date"])] += 1
            if split == "train" and frame_no[(lot, i["date"])] % 2 == 0:
                continue
            by_set[f"pklot_{split}"].append(i)
    return by_set


def write_pklot(by_set: dict[str, list[dict]], workers: int) -> dict:
    report = {}
    for name in ["pklot_train", "pklot_val", "pklot_test", "pklot_unseen"]:
        items = by_set[name]
        writer, frames, skipped = PatchWriter(name), 0, Counter()
        with ProcessPoolExecutor(workers) as pool:
            for item, out, err in pool.map(_cut_pklot, items, chunksize=16):
                if err:
                    skipped[err.split(":")[0]] += 1
                    continue
                p, labels = out
                rel = item["path"].relative_to(PKLOT_DIR).as_posix()
                writer.add(p, [{"label": occ, "dataset": "pklot", "camera": item["lot"],
                                "weather": item["weather"], "date": item["date"], "image": rel, "slot": sid}
                               for sid, occ in labels])
                frames += 1
        writer.close()
        labels = np.array([m["label"] for m in writer.meta])
        report[name] = {"frames": frames, "patches": len(labels), "occupied": int(labels.sum()),
                        "skipped_frames": dict(skipped)}
        print(name, report[name], flush=True)
    return report


def _read_cnr_patch(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_cnr(workers: int) -> dict:
    weather_name = {"S": "sunny", "O": "overcast", "R": "rainy"}
    rows = []
    with open(CNR_DIR / "CNRPark+EXT.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["image_url"].startswith("CNR-EXT/"):
                rows.append(r)
    rows.sort(key=lambda r: r["image_url"])
    paths = [CNR_DIR / r["image_url"].removeprefix("CNR-EXT/") for r in rows]
    writer = PatchWriter("cnr_ext")
    with ProcessPoolExecutor(workers) as pool:
        for r, patch in zip(rows, pool.map(_read_cnr_patch, paths, chunksize=256)):
            date, time = r["datetime"].split("_")
            writer.add(patch[None], [{"label": int(r["occupancy"]), "dataset": "cnr_ext",
                                      "camera": f"camera{int(r['camera'])}", "weather": weather_name[r["weather"]],
                                      "date": date, "time": time, "image": r["image_url"],
                                      "slot": int(r["slot_id"])}])
    writer.close()
    labels = np.array([m["label"] for m in writer.meta])
    report = {"patches": len(labels), "occupied": int(labels.sum())}
    print("cnr_ext", report, flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--only", choices=["pklot", "cnr"], default=None)
    args = p.parse_args()

    out = ROOT / "results" / "splits.json"
    splits = json.loads(out.read_text()) if out.exists() else {}

    if args.only in (None, "pklot"):
        items = pklot_images()
        if not items:
            sys.exit(f"no PKLot images under {PKLOT_DIR}; run scripts/download_data.sh first")
        by_set = build_pklot_sets(items, args.workers)
        days = defaultdict(lambda: defaultdict(set))
        for name, its in by_set.items():
            for i in its:
                days[i["lot"]][name.removeprefix("pklot_")].add(i["date"])
        splits["pklot_days"] = {lot: {s: sorted(d) for s, d in sorted(v.items())} for lot, v in sorted(days.items())}
        splits["pklot_sets"] = write_pklot(by_set, args.workers)

    if args.only in (None, "cnr"):
        splits["cnr_ext"] = write_cnr(args.workers)
        dates = sorted({p.name for p in (CNR_DIR / "PATCHES").glob("*/*")})
        splits["cnr_days"] = split_cnr_days(dates)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
