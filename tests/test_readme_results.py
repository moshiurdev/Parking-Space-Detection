"""Guard against the README drifting away from the measured results."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("readme_tables", ROOT / "scripts" / "readme_tables.py")
readme_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readme_tables)

HAS_RESULTS = (ROOT / "results" / "mobilenetv3" / "metrics.json").exists()


@pytest.mark.skipif(not HAS_RESULTS, reason="no evaluation results yet")
@pytest.mark.parametrize("table", ["main", "weather", "gap", "latency", "camera"])
def test_readme_tables_match_metrics(table):
    readme = (ROOT / "README.md").read_text()
    rows = readme_tables.all_rows()[table]
    assert rows, f"no results for table {table!r}"
    for row in rows:
        assert row in readme, f"README is missing or has a stale row:\n{row}"


@pytest.mark.skipif(not HAS_RESULTS, reason="no evaluation results yet")
def test_readme_headline_numbers():
    readme = (ROOT / "README.md").read_text()
    s = readme_tables.load("mobilenetv3")["sets"]
    for name in ["pklot_test", "pklot_unseen", "cnr_ext"]:
        assert readme_tables.pct(s[name]["overall"]["accuracy"]) in readme.split("## Features")[0]


@pytest.mark.skipif(not HAS_RESULTS, reason="no evaluation results yet")
def test_readme_error_analysis_numbers():
    readme = (ROOT / "README.md").read_text()
    (_, fp), (fn, _) = readme_tables.load("mobilenetv3")["sets"]["cnr_ext"]["overall"]["confusion_matrix"]
    assert f"{fn:,} occupied slots are predicted empty and only {fp:,} empty slots" in readme
