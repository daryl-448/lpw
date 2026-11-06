"""Build PILOT_conf/: the same slice, with a fixed 20% of TRAINING groups (seed 0) relabelled split = 'conf'.
These untouched confirmation groups are never trained on, early-stopped on, recalibrated on or selected on
(reports/p1_base_qualification.md). The other files are symlinked unchanged: venues, hubs and groups are label-blind
structure built over all agents before the split (lpw.data.slice), exactly as for val and test.
usage: make_conf_pilot.py PILOT"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

FRAC, SEED = 0.2, 0
src = Path(sys.argv[1]).resolve()
dst = src.parent / (src.name + "_conf")
dst.mkdir(exist_ok=True)
a = pl.read_parquet(src / "agents.parquet")
tg = np.unique(a.filter(pl.col("split") == "train")["group_id"].to_numpy())
conf = np.random.default_rng(SEED).choice(tg, int(round(FRAC * len(tg))), replace=False)
a2 = a.with_columns(pl.when(pl.col("group_id").is_in(conf.tolist()) & (pl.col("split") == "train")).then(pl.lit("conf"))
                    .otherwise(pl.col("split")).alias("split"))
a2.write_parquet(dst / "agents.parquet")
for f in src.iterdir():
    if f.name != "agents.parquet" and not (dst / f.name).exists():
        (dst / f.name).symlink_to(Path("..") / src.name / f.name)  # relative: the tree may be synced elsewhere
counts = a2.group_by("split", "is_scam").len().sort("split", "is_scam").to_dicts()
(dst / "conf.json").write_text(json.dumps({"source": str(src), "frac": FRAC, "seed": SEED, "conf_groups": len(conf),
                                           "train_groups_before": len(tg), "counts": counts}, indent=1))
print(dst, json.dumps(counts))
