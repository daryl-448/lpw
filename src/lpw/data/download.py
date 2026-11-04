"""Download the pilot files from the HuggingFace dataset, with resume and sha256 checks.

Usage: uv run python -m lpw.data.download [--dest data/raw] [--months 2021-05:2022-04]
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "fesevu/ethereum_fraud_dataset_by_activity"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
UA = {"User-Agent": "lpw-downloader"}

STATIC_FILES = [
    "gnn_dataset/labels/targets_global.parquet",
    "gnn_dataset/mapping/address_id_map_labels.parquet",
    "gnn_dataset/meta/month_window_meta.parquet",
    "gnn_dataset/targets/month_targets.parquet",
    "addr_labels_balanced.csv.zst",
    "lstm_dataset/daily_filtered.parquet",
]


def month_range(spec: str) -> list[str]:
    start, end = spec.split(":")
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def listing(subdir: str, recursive: bool = True) -> dict[str, dict]:
    url = f"{API}/{subdir}" + ("?recursive=true" if recursive else "")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return {f["path"]: f for f in json.load(r) if f.get("type") == "file"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(path: str, meta: dict, dest: Path) -> None:
    out = dest / path
    out.parent.mkdir(parents=True, exist_ok=True)
    size = meta["size"]
    want = meta.get("lfs", {}).get("oid")
    if out.exists() and out.stat().st_size == size and (want is None or sha256(out) == want):
        print(f"ok (cached) {path}")
        return
    part = out.with_suffix(out.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if 0 < have < size:
        headers["Range"] = f"bytes={have}-"
    else:
        have = 0
    req = urllib.request.Request(f"{RESOLVE}/{path}", headers=headers)
    with urllib.request.urlopen(req) as r, part.open("ab" if have else "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    if part.stat().st_size != size:
        sys.exit(f"size mismatch for {path}: {part.stat().st_size} != {size}")
    if want is not None and sha256(part) != want:
        part.unlink()
        sys.exit(f"sha256 mismatch for {path}")
    part.rename(out)
    print(f"ok {path} ({size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--months", default="2021-05:2022-04")
    args = ap.parse_args()
    dest = Path(args.dest)
    files = {}
    for sub in ["gnn_dataset/labels", "gnn_dataset/mapping", "gnn_dataset/meta", "gnn_dataset/targets", "lstm_dataset"]:
        files.update(listing(sub))
    files.update(listing("", recursive=False))
    months = listing("gnn_dataset/edges_by_month")
    wanted = list(STATIC_FILES) + [f"gnn_dataset/edges_by_month/month={m}/edges.parquet" for m in month_range(args.months)]
    files.update(months)
    for p in wanted:
        if p not in files:
            sys.exit(f"not in HF listing: {p}")
        fetch(p, files[p], dest)


if __name__ == "__main__":
    main()
