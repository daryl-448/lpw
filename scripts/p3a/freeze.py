"""Freeze the P3a population models (reports/p3a_prereg.md §2). Run once, before any P3a readout.

usage: uv run python scripts/p3a/freeze.py [--out runs/p3a/2018/population]
"""
import argparse
import json
from pathlib import Path

from lpw.p1.store import build_store
from lpw.p3.population import freeze

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/p3a/2018/population")
a = ap.parse_args()
meta = freeze(build_store(Path("data/pilot_2018")), Path(a.out), log=lambda *x: None)
print(json.dumps(meta, indent=1, default=float))
