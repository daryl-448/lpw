"""Fully synthetic populations for P3a (reports/p3a_prereg.md §2).

Per agent, in order: role ~ Bernoulli(pi_train); cohort, activation time, raw profile and initial balance from the
frozen ProfileGen; the plan s = (count, amount bucket) from the frozen default PlanPrior. Neither reads test agents. Val
is used only for ProfileGen's early stopping (its epoch is chosen by val NLL, roles included) and PlanPrior's
reported val NLL.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from lpw.p1.plan import PlanPrior
from lpw.p1.profile_gen import ProfileGen

PG_SEED = 0          # ProfileGen training seed (its default)
PRIOR_SEED = 0       # PlanPrior seed (its default)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(store, out: Path, dev="cuda", log=print) -> dict:
    """Fit the population models on the train split once and persist them. Refuses to overwrite."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pg_p, pr_p, meta_p = out / "profile_gen.pt", out / "plan_prior.pkl", out / "population.json"
    assert not meta_p.exists(), f"{meta_p} exists: the population models are frozen"
    pg = ProfileGen(store, dev=dev, seed=PG_SEED, log=log)
    pg.save(pg_p)
    prior = PlanPrior(store, seed=PRIOR_SEED)
    pr_p.write_bytes(pickle.dumps(prior))
    tr = store.agents("train")
    meta = {"pi_train": float(store.role[tr].mean()), "n_train": int(len(tr)), "n_val": int(len(store.agents("val"))),
            "p_born": pg.p_born, "profile_gen_val_nll": pg.val_nll, "profile_gen_epochs": pg.epochs,
            "plan_prior_val_nll": prior.val_nll, "sha256": {"profile_gen.pt": sha(pg_p), "plan_prior.pkl": sha(pr_p)}}
    meta_p.write_text(json.dumps(meta, indent=1, default=float))
    return meta


class Population:
    """The frozen population models, loaded with their hashes checked."""

    def __init__(self, store, root: Path, dev="cuda"):
        root = Path(root)
        self.meta = json.loads((root / "population.json").read_text())
        for f, h in self.meta["sha256"].items():
            assert sha(root / f) == h, f"{root / f} changed since freeze"
        self.store = store
        self.pg = ProfileGen.load(root / "profile_gen.pt", store, dev=dev)
        self.prior = pickle.loads((root / "plan_prior.pkl").read_bytes())
        self.pi = self.meta["pi_train"]

    def sample(self, n: int, seed: int):
        """n agents. Returns (conds for lpw.p1.generate, raw profiles (n, 20)). Each call with the same seed returns the
        same population."""
        rng = np.random.default_rng(seed)
        roles = (rng.random(n) < self.pi).astype(np.int64)
        conds, raw = self.pg.population(roles, rng, return_raw=True)
        pn, pa = self.prior.sample(roles, np.array([c["cohort"] for c in conds]), np.stack([c["prof"] for c in conds]), rng)
        for c, a, b in zip(conds, pn, pa):
            c["plan_n"], c["plan_a"] = int(a), int(b)
        return conds, raw
