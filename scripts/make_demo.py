"""Build the frame sequences that the web demo replays.

Writes ``data/demo/sequences.json``. Frames are referenced in place inside
``data/raw/`` (nothing is copied). Only test data is used:

* PKLot UFPR05: the lot that was never used for training (one full day)
* PKLot PUCPR: one held-out test day of a training lot
* CNRPark-EXT: one camera over several days after the fine-tuning days
  (it takes one frame every 30 minutes)

Usage:
    python scripts/make_demo.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parking.slots import read_cnr_camera_csv, read_pklot_xml  # noqa: E402

PKLOT = ROOT / "data" / "raw" / "pklot" / "PKLot" / "PKLot"
CNR = ROOT / "data" / "raw" / "cnr"
OUT = ROOT / "data" / "demo" / "sequences.json"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def pklot_day(lot: str, date: str, max_frames: int) -> list[dict]:
    frames = []
    for jpg in sorted(PKLOT.glob(f"{lot}/*/{date}/*.jpg"), key=lambda p: p.name)[:max_frames]:
        xml = jpg.with_suffix(".xml")
        if not xml.exists():
            continue
        try:
            slots = read_pklot_xml(xml)
        except Exception:
            continue
        hh, mm = jpg.stem.split("_")[1:3]
        frames.append({
            "image": rel(jpg), "time": f"{date} {hh}:{mm}", "weather": jpg.parts[-3].lower(),
            "slots": [{"id": s["id"], "polygon": s["polygon"].round(1).tolist(), "label": s["occupied"]}
                      for s in slots],
        })
    return frames


def cnr_days(camera: int, dates: list[str], max_frames: int) -> list[dict]:
    boxes = read_cnr_camera_csv(CNR / f"camera{camera}.csv")
    labels = defaultdict(dict)  # time -> slot -> label
    with open(CNR / "CNRPark+EXT.csv", newline="") as f:
        for r in csv.DictReader(f):
            date, time = r["datetime"].split("_")
            if r["image_url"].startswith("CNR-EXT/") and int(r["camera"]) == camera and date in dates:
                labels[(date, time.replace(".", ""))][int(r["slot_id"])] = int(r["occupancy"])
    frames = []
    jpgs = [j for d in dates for j in CNR.glob(f"FULL_IMAGE_1000x750/*/{d}/camera{camera}/*.jpg")]
    for jpg in sorted(jpgs, key=lambda p: p.name):
        date, t = jpg.stem.split("_")
        if (date, t) not in labels:
            continue  # frame without annotations
        lab = labels[(date, t)]
        frames.append({
            "image": rel(jpg), "time": f"{date} {t[:2]}:{t[2:]}", "weather": jpg.parts[-4].lower(),
            "slots": [{"id": sid, "polygon": poly.round(1).tolist(), "label": lab.get(sid)}
                      for sid, poly in sorted(boxes.items())],
        })
    return frames[:max_frames]


def main() -> None:
    splits = json.loads((ROOT / "results" / "splits.json").read_text())
    pucpr_test = splits["pklot_days"]["PUCPR"]["test"]
    ufpr05_days = splits["pklot_days"]["UFPR05"]["unseen"]
    later = [d for d, s in splits["cnr_days"].items() if s == "later"]

    def n_frames(lot, d):
        return len(list(PKLOT.glob(f"{lot}/*/{d}/*.jpg")))

    # the most complete day of each lot, ties broken by date
    ufpr05_day = max(ufpr05_days, key=lambda d: (n_frames("UFPR05", d), d))
    pucpr_day = max(pucpr_test, key=lambda d: (n_frames("PUCPR", d), d))
    cnr_dates, cnr_cam = sorted(later)[:6], 8

    sequences = [
        {"id": "ufpr05", "title": f"PKLot UFPR05, {ufpr05_day}",
         "description": "Unseen lot: this camera and lot were never used for training or model selection.",
         "frames": pklot_day("UFPR05", ufpr05_day, 150)},
        {"id": "pucpr", "title": f"PKLot PUCPR, {pucpr_day}",
         "description": "Training lot, held-out test day (about 100 slots per frame).",
         "frames": pklot_day("PUCPR", pucpr_day, 150)},
        {"id": "cnr", "title": f"CNRPark-EXT camera {cnr_cam}, {cnr_dates[0]} to {cnr_dates[-1]}",
         "description": "Different dataset (Pisa, Italy): cross-dataset test of a model trained only on PKLot.",
         "frames": cnr_days(cnr_cam, cnr_dates, 150)},
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sequences": sequences}))
    for s in sequences:
        print(f"{s['id']}: {s['title']}, {len(s['frames'])} frames")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
