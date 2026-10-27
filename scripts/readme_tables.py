"""Print the README result tables from ``results/*/metrics.json``.

The README tables are pasted from this output, and ``tests/test_readme_results.py``
checks that every row printed here appears in the README verbatim, so the
README cannot drift from the measured numbers.

Usage:
    python scripts/readme_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELS = [("malexnet", "mAlexNet (from scratch)"), ("mobilenetv3", "**MobileNetV3-small (ImageNet)**")]
WEATHER_RUNS = [("mobilenetv3", "all weather (sunny, cloudy, rainy)"), ("mobilenetv3_sunny", "sunny frames only")]
GAP_RUNS = [("mobilenetv3", "MobileNetV3-small, standard augmentation"),
            ("mobilenetv3_strong", "+ strong augmentation (PKLot only)"),
            ("mobilenetv3_cnr_ft", "+ fine-tuning on the first 3 CNR-EXT days")]
PKLOT_WEATHER = ["sunny", "cloudy", "rainy"]
CNR_WEATHER = ["sunny", "overcast", "rainy"]
LATENCY_FRAMES = [("PKLot PUCPR", "PUCPR, 100 slots"), ("PKLot UFPR05", "UFPR05, 40 slots"),
                  ("CNR-EXT camera1", "CNR-EXT camera 1, 35 slots")]


def load(run: str) -> dict | None:
    path = RESULTS / run / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def cell(s: dict) -> str:
    return f"{pct(s['accuracy'])} / {s['macro_f1']:.4f}"


def main_rows() -> list[str]:
    rows = []
    for run, label in MODELS:
        m = load(run)
        if m:
            s = m["sets"]
            ms = m["cpu_latency_ms"]["onnxruntime"]["PKLot PUCPR"]["total_ms"]
            rows.append(f"| {label} | {m['params'] / 1e6:.2f} M | {cell(s['pklot_test']['overall'])} | "
                        f"{cell(s['pklot_unseen']['overall'])} | {cell(s['cnr_ext']['overall'])} | {ms:.0f} ms |")
    return rows


def weather_rows() -> list[str]:
    rows = []
    for run, label in WEATHER_RUNS:
        m = load(run)
        if m:
            s = m["sets"]
            cells = [pct(s[name]["by_weather"][w]["accuracy"])
                     for name, ws in [("pklot_test", PKLOT_WEATHER), ("pklot_unseen", PKLOT_WEATHER),
                                      ("cnr_ext", CNR_WEATHER)] for w in ws]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def gap_rows() -> list[str]:
    rows = []
    for run, label in GAP_RUNS:
        m = load(run)
        if m:
            s = m["sets"]
            later = s["cnr_ext_later"]
            w = later["by_weather"]
            rows.append(f"| {label} | {cell(s['pklot_test']['overall'])} | {cell(s['pklot_unseen']['overall'])} | "
                        f"{cell(later['overall'])} | {pct(w['sunny']['accuracy'])} | "
                        f"{pct(w['overcast']['accuracy'])} | {pct(w['rainy']['accuracy'])} |")
    return rows


def latency_rows() -> list[str]:
    rows = []
    for run, label in MODELS:
        m = load(run)
        if m:
            lat = m["cpu_latency_ms"]
            cells = [f"{lat['onnxruntime'][k]['total_ms']:.1f} ms ({lat['pytorch'][k]['total_ms']:.0f} ms)"
                     for k, _ in LATENCY_FRAMES]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def camera_rows(run: str = "mobilenetv3") -> list[str]:
    m = load(run)
    if not m:
        return []
    cams = m["sets"]["cnr_ext"]["by_camera"]
    order = sorted(cams, key=lambda c: int(c.removeprefix("camera")))
    return ["| " + " | ".join(c.removeprefix("camera") for c in order) + " |",
            "| " + " | ".join(pct(cams[c]["accuracy"]) for c in order) + " |"]


def all_rows() -> dict[str, list[str]]:
    return {"main": main_rows(), "weather": weather_rows(), "gap": gap_rows(),
            "latency": latency_rows(), "camera": camera_rows()}


if __name__ == "__main__":
    for name, rows in all_rows().items():
        print(f"## {name}")
        print("\n".join(rows))
        print()
