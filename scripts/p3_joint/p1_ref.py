"""Three-panel P1 reference for kct / B on val (reports/p3_joint_addendum.md §J7), made outside the joint entry point.

lpw.p1.evaluate.evaluate is called directly on B with the scripts/p1_audit/b1gap.py loading recipe (plan_inference
False) on the guarded ("train", "val") store. Its 'all' panel must match the archived aggregate
runs/p1_audit/2018/base_plan_prog_rp_val_eval.json within 1e-4 in every field.

usage: uv run python scripts/p3_joint/p1_ref.py   -> runs/p3_joint/2018/p1_ref/B_val_eval.json (refuses to overwrite)
"""
import json
from pathlib import Path

from lpw.p1.batch import Views
from lpw.p1.evaluate import evaluate
from lpw.p1.store import build_store
from lpw.p1.train import load

PILOT = Path("data/pilot_2018")
OUT = Path("runs/p3_joint/2018/p1_ref/B_val_eval.json")
ARCHIVED = Path("runs/p1_audit/2018/base_plan_prog_rp_val_eval.json")
assert not OUT.exists(), f"{OUT} exists"
st = build_store(PILOT, splits=("train", "val"))
m = load(PILOT, "lpw_plan_prog", Path("runs/p1_r0/2018/lpw_plan_prog_rp"), st)
m.plan_inference = False
ev = evaluate(m, st, "val", views=Views(st))
arch = json.loads(ARCHIVED.read_text())
diff = {k: abs(ev["all"][k] - v) for k, v in arch.items()}
bad = {k: d for k, d in diff.items() if d > 1e-4}
assert not bad, f"guarded 'all' panel differs from the archived aggregate: {bad}"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(ev, indent=1))
print(f"wrote {OUT}; max |all - archived| = {max(diff.values()):.2e} over {len(diff)} fields")
