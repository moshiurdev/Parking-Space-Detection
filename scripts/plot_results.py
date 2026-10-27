"""Plot the domain-shift results of all evaluated runs.

Reads ``results/<run>/metrics.json`` and writes ``results/domain_shift.png``:
(left) error rate of every model on the three test sets, (right) error rate by
weather on the unseen lot and on CNR-EXT for the all-weather and sunny-only models.
Error rate (100 - accuracy) is plotted so that bars can start at zero and the
small in-domain differences stay visible.

Usage:
    python scripts/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RUNS = [("malexnet", "mAlexNet\n(baseline)"), ("mobilenetv3", "MobileNetV3-S"),
        ("mobilenetv3_strong", "MobileNetV3-S\nstrong aug."), ("mobilenetv3_cnr_ft", "MobileNetV3-S\n+ CNR fine-tune")]
SETS = [("pklot_test", "PKLot held-out days"), ("pklot_unseen", "PKLot unseen lot"),
        ("cnr_ext_later", "CNRPark-EXT (later days)")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, groups, series, values):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, color) in enumerate(zip(series, COLORS)):
        v = np.array([values[g][i] for g in range(len(groups))], dtype=float)
        xs = x + (i - (n - 1) / 2) * width
        ax.bar(xs, v, width * 0.92, color=color, label=name)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi, f"{vi:.1f}" if vi >= 1 else f"{vi:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, fontsize=8.5, color=INK)
    ax.set_ylim(0, max(max(v) for v in values) * 1.12)
    ax.set_ylabel("error rate (%), lower is better", color=MUTED)
    style(ax)


def main() -> None:
    metrics = {r: json.loads((RESULTS / r / "metrics.json").read_text())
               for r, _ in RUNS + [("mobilenetv3_sunny", "")] if (RESULTS / r / "metrics.json").exists()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})

    runs = [(r, label) for r, label in RUNS if r in metrics]
    vals = [[100 * (1 - metrics[r]["sets"][s]["overall"]["accuracy"]) for s, _ in SETS] for r, _ in runs]
    bars(a, [label for _, label in runs], [t for _, t in SETS], vals)
    a.set_title("Error rate by test set", fontsize=10, color=INK, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.32))

    groups = [("pklot_unseen", w, f"UFPR05\n{w}") for w in ["sunny", "cloudy", "rainy"]] + \
             [("cnr_ext", w, f"CNR-EXT\n{w}") for w in ["sunny", "overcast", "rainy"]]
    wruns = [(r, label) for r, label in [("mobilenetv3", "all weather"), ("mobilenetv3_sunny", "sunny only")]
             if r in metrics]
    wvals = [[100 * (1 - metrics[r]["sets"][name]["by_weather"][w]["accuracy"]) for r, _ in wruns]
             for name, w, _ in groups]
    bars(b, [g for _, _, g in groups], [f"trained on {label} PKLot" for _, label in wruns], wvals)
    b.set_title("Unseen lot and CNR-EXT by weather (MobileNetV3-S)", fontsize=10, color=INK, loc="left")
    b.legend(frameon=False, fontsize=8, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.32))

    fig.tight_layout()
    out = RESULTS / "domain_shift.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
