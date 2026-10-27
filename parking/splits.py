"""Day-level splits.

Consecutive frames of a camera are 5 minutes apart and nearly identical, so a
random patch-level split would put almost the same image in train and test.
Splits are therefore made by whole days (and whole lots / cameras for the
cross-domain tests).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

SOURCE_LOTS = ["PUCPR", "UFPR04"]   # training lots (in-domain)
UNSEEN_LOT = "UFPR05"               # never used for training or model selection
CNR_ADAPT_DAYS = 3                  # first CNR-EXT days used for the fine-tuning experiment


def split_days(day_weather: dict[str, list[str]], val_frac: float = 0.15, test_frac: float = 0.25,
               seed: int = 0) -> dict[str, str]:
    """Assign each day of one lot to ``train``, ``val`` or ``test``.

    ``day_weather`` maps a date to the weather label of each of its images.
    Days are stratified by their majority weather, so every weather condition
    that has at least three days appears in all three splits.
    """
    rng = np.random.default_rng(seed)
    by_weather = defaultdict(list)
    for day in sorted(day_weather):
        majority = Counter(day_weather[day]).most_common(1)[0][0]
        by_weather[majority].append(day)
    assignment = {}
    for weather in sorted(by_weather):
        days = list(rng.permutation(by_weather[weather]))
        n = len(days)
        n_test = max(1, round(test_frac * n)) if n >= 2 else 0
        n_val = max(1, round(val_frac * n)) if n >= 3 else 0
        for i, day in enumerate(days):
            assignment[str(day)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assignment


def split_cnr_days(days: list[str], n_adapt: int = CNR_ADAPT_DAYS) -> dict[str, str]:
    """Chronological split of CNR-EXT: the first ``n_adapt`` days are the
    labelled "new site" data for fine-tuning, the rest are the test days."""
    ordered = sorted(set(days))
    return {d: ("adapt" if i < n_adapt else "later") for i, d in enumerate(ordered)}
