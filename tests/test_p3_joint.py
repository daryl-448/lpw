"""Fixtures for the joint sealed-test entry point (reports/p3_joint_addendum.md §J7). No real test rows are touched:
authorization runs against throwaway git repositories, and the statistics against fabricated data."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lpw import sealed
from lpw.p3.utility import GroupBoot, transfer_contrast

REC = sealed.RECORD
MAN = sealed.MANIFEST


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()


def make_repo(tmp_path: Path, record_commit=None, extra_in_auth=False, write_record=True) -> Path:
    """A frozen commit (code + manifest) followed by an authorization-only child commit."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@example.org")
    git(r, "config", "user.name", "t")
    (r / ".gitignore").write_text("runs/\n")
    (r / "code.py").write_text("x = 1\n")
    (r / "reports").mkdir()
    (r / MAN).write_text(json.dumps({"sha256": {"code.py": sealed.sha256(r / "code.py")}}))
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "freeze")
    frozen = git(r, "rev-parse", "HEAD")
    if write_record:
        (r / REC).write_text(f"# authorization\n\nAuthorized commit: {record_commit or frozen}\n")
    if extra_in_auth:
        (r / "other.txt").write_text("sneaky\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "--allow-empty", "-m", "authorize")
    return r


@pytest.fixture(autouse=True)
def _reset_authorization():
    sealed._AUTHORIZED = None
    yield
    sealed._AUTHORIZED = None


def test_good_authorization_passes_and_unlocks_test(tmp_path):
    r = make_repo(tmp_path)
    assert not sealed.allowed("test")
    frozen = sealed.authorize(repo=r)
    assert frozen == git(r, "rev-parse", "HEAD^") and sealed.allowed("test")


def test_absent_record_is_refused(tmp_path):
    r = make_repo(tmp_path, write_record=False)
    with pytest.raises(sealed.SealedError, match="no authorization record"):
        sealed.check(repo=r)


def test_record_naming_the_wrong_commit_is_refused(tmp_path):
    r = make_repo(tmp_path, record_commit="0" * 40)
    with pytest.raises(sealed.SealedError, match="parents"):
        sealed.check(repo=r)


def test_authorization_commit_touching_another_file_is_refused(tmp_path):
    r = make_repo(tmp_path, extra_in_auth=True)
    with pytest.raises(sealed.SealedError, match="must change only"):
        sealed.check(repo=r)


def test_a_commit_after_the_authorization_is_refused(tmp_path):
    r = make_repo(tmp_path)
    (r / "later.txt").write_text("x\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "later")
    with pytest.raises(sealed.SealedError, match="parents"):
        sealed.check(repo=r)


def test_dirty_tree_is_refused(tmp_path):
    r = make_repo(tmp_path)
    (r / "untracked.txt").write_text("x\n")
    with pytest.raises(sealed.SealedError, match="not clean"):
        sealed.check(repo=r)


def test_changed_hash_is_refused(tmp_path):
    r = make_repo(tmp_path)
    man = json.loads((r / MAN).read_text())
    man["sha256"]["code.py"] = "0" * 64
    (r / MAN).write_text(json.dumps(man))
    git(r, "add", "-A")
    git(r, "commit", "-q", "--amend", "--no-edit")  # keep a clean tree; the parent check still passes
    with pytest.raises(sealed.SealedError):
        sealed.check(repo=r)


def test_existing_output_or_staging_is_refused(tmp_path):
    for d in (sealed.TEST_OUT, sealed.staging_dir(sealed.TEST_OUT)):
        (tmp_path / d.name).mkdir()
        r = make_repo(tmp_path / d.name)
        (r / d).mkdir(parents=True)  # runs/ is ignored, so the tree stays clean
        with pytest.raises(sealed.SealedError, match="already exists"):
            sealed.check(repo=r)


def test_development_guards_refuse_test_without_authorization():
    from lpw.p3b.access import load
    with pytest.raises(sealed.SealedError):
        load(("train", "test"))
    with pytest.raises(sealed.SealedError):
        sealed.require("test")


def test_publish_is_atomic_and_an_interrupted_run_publishes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "test"
    st = sealed.staging_dir(out)
    st.mkdir()
    (st / "a.json").write_text("{}")
    monkeypatch.setattr(sealed.os, "rename", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        sealed.publish(st, out)
    assert not out.exists() and st.exists()  # nothing published; the staging directory is kept for §J8.4
    monkeypatch.undo()
    man = sealed.publish(st, out)
    assert out.exists() and not st.exists() and set(man["sha256"]) == {"a.json"}
    assert json.loads((out / "output_manifest.json").read_text()) == man


def test_transfer_contrast_is_invalid_below_the_valid_draw_minimum():
    rng = np.random.default_rng(0)
    y = np.zeros(400, int)
    y[:3] = 1  # three scams in three groups: most whole-group draws miss every scam
    groups = np.arange(400) // 4
    boot = GroupBoot(groups, B=1000)
    s = y + rng.normal(0, 0.1, len(y))
    r = transfer_contrast(y, [s, s], s, boot)
    assert r["n_valid_draws"] < 950 and r["outcome"] == "INVALID" and r["D_lower_1s95"] is None


def test_transfer_contrast_gate_precedes_validity():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    noise = rng.random(len(y))
    r = transfer_contrast(y, [noise], noise, GroupBoot(np.arange(2000) // 5, B=200))
    assert r["outcome"] == "NOT TESTABLE" and r["n_valid_draws"] == 200


def test_giant_group_bootstrap_counts_valid_draws():
    rng = np.random.default_rng(2)
    y = (rng.random(1000) < 0.1).astype(int)
    groups = np.where(np.arange(1000) < 600, 0, np.arange(1000))  # one group holds 60% of the agents
    s = y + rng.normal(0, 0.5, len(y))
    r = transfer_contrast(y, [s], s, GroupBoot(groups, B=300))
    assert r["n_draws"] == 300 and r["n_valid_draws"] == 300 and r["outcome"] in ("SUPPORTED", "CONSISTENT")


def test_incremental_one_role_set_is_not_testable():
    from lpw.p3b import evaluate as E
    y = np.zeros(50, int)
    s = np.random.default_rng(0).random(50)
    out = E.incremental(y, [np.arange(50)] * 10, s, s, [s], {"p3b0": [s], "p3b1": [s]})
    assert out["point"] is None and out["n_valid_draws"] == 0


def test_matcher_below_200_pairs_is_not_testable():
    import polars as pl

    from lpw.p3.matching import MIN_PAIRS, Matcher
    rng = np.random.default_rng(3)
    cols = ["p_age_days", "p_days_active", "p_days_since_active", "p_normal_sent_cnt", "p_normal_recv_cnt", "p_erc20_sent_cnt",
            "p_erc20_recv_cnt", "p_eth_sent_sum", "p_eth_recv_sum", "act_frac"]

    def F(n):  # the frozen confounder columns, as in tests/test_p3a.py
        y = (rng.random(n) < 0.3).astype(int)
        return pl.DataFrame({**{c: rng.normal(size=n) for c in cols}, "cohort": (rng.random(n) < 0.5).astype(float)}), y
    r = Matcher(*F(3000)).match(*F(300))
    assert MIN_PAIRS == 200 and r["n_pairs"] < MIN_PAIRS and not r["testable"]


def test_invalid_contrast_formats_and_reports_the_scam_minimum():
    from lpw.p3.utility import contrast_line
    rng = np.random.default_rng(0)
    y = np.zeros(400, int)
    y[:3] = 1
    s = y + rng.normal(0, 0.1, len(y))
    r = transfer_contrast(y, [s, s], s, GroupBoot(np.arange(400) // 4, B=1000))
    assert r["outcome"] == "INVALID" and r["min_scam_per_draw"] == 0
    assert "INVALID" in contrast_line(r) and "CI95 n/a" in contrast_line(r)  # the headline line must not crash


def test_empty_and_one_role_subsets_are_not_testable():
    from lpw.p3.utility import contrast_line
    empty = np.array([], int)
    r = transfer_contrast(empty, [empty.astype(float)], empty.astype(float), GroupBoot(empty, B=10))
    assert r["outcome"] == "NOT TESTABLE" and r["n_draws"] == 10 and r["min_scam_per_draw"] == 0
    y = np.zeros(50, int)
    s = np.random.default_rng(0).random(50)
    r = transfer_contrast(y, [s], s, GroupBoot(np.arange(50) // 5, B=10))
    assert r["outcome"] == "NOT TESTABLE" and r["reason"] == "one role or empty" and "NOT TESTABLE" in contrast_line(r)


def test_test_mode_refuses_artifact_root_overrides():
    import os
    import sys
    env = {**os.environ, "P3A_ROOT": "/elsewhere"}
    p = subprocess.run([sys.executable, "scripts/p3_joint/run.py", "--split", "test"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "artifact-root overrides" in p.stderr and not sealed.staging_dir(sealed.TEST_OUT).exists()
