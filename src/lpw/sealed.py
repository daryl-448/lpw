"""Sealed-test authorization and atomic publication (reports/p3_joint_addendum.md §J7-§J8).

The 2018 test split is reachable only after `authorize()` has passed, in this process. It checks:
  1. HEAD is an authorization-only commit: its single parent is the frozen commit named in the authorization record,
     and HEAD^..HEAD changes only that record;
  2. the working tree is clean (no tracked change, no untracked non-ignored file);
  3. every sha256 in the frozen manifest matches;
  4. neither the output directory nor its staging directory exists.
Development code asks `allowed(split)`: train and val always, test only after authorize().

Results are written into a staging directory and published by one atomic rename (`publish`), after every file and a
hash manifest of them are fsynced. A run that ends without the rename publishes nothing.
"""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

DEV_SPLITS = ("train", "val")
RECORD = Path("reports/p3_joint_authorization.md")
MANIFEST = Path("reports/p3_joint_manifest.json")
TEST_OUT = Path("runs/p3_joint/2018/test")

_AUTHORIZED: str | None = None  # the authorized freeze commit, once authorize() has passed in this process


class SealedError(RuntimeError):
    pass


def allowed(split: str) -> bool:
    return split in DEV_SPLITS or (split == "test" and _AUTHORIZED is not None)


def require(split: str) -> None:
    if not allowed(split):
        raise SealedError(f"split {split!r}: the sealed test split is opened once, with authorization (lpw.sealed)")


def staging_dir(out: Path) -> Path:
    return out.parent / f".{out.name}_staging"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def authorized_commit(record_text: str) -> str:
    m = re.findall(r"^Authorized commit:\s*([0-9a-f]{40})\s*$", record_text, re.M)
    if len(m) != 1:
        raise SealedError("the authorization record must name exactly one 'Authorized commit: <40-hex sha>'")
    return m[0]


def check(repo: Path = Path("."), record: Path = RECORD, manifest: Path = MANIFEST, out: Path = TEST_OUT) -> str:
    """All four authorization checks; returns the authorized freeze commit or raises SealedError."""
    repo = Path(repo)
    rec = repo / record
    if not rec.exists():
        raise SealedError(f"no authorization record at {record}")
    frozen = authorized_commit(rec.read_text())
    parents = _git(repo, "rev-list", "--parents", "-n", "1", "HEAD").split()[1:]
    if parents != [frozen]:
        raise SealedError(f"HEAD's parents {parents} are not exactly the authorized freeze commit {frozen}")
    changed = [x for x in _git(repo, "diff", "--name-only", "HEAD^", "HEAD").splitlines() if x]
    if changed != [record.as_posix()]:
        raise SealedError(f"the authorization commit must change only {record}; it changes {changed}")
    if _git(repo, "status", "--porcelain").strip():
        raise SealedError("the working tree is not clean")
    man = repo / manifest
    if not man.exists():
        raise SealedError(f"no frozen manifest at {manifest}")
    bad = [p for p, h in json.loads(man.read_text())["sha256"].items() if not (repo / p).exists() or sha256(repo / p) != h]
    if bad:
        raise SealedError(f"{len(bad)} manifest hash mismatches, e.g. {bad[:5]}")
    for d in (out, staging_dir(out)):
        if (repo / d).exists():
            raise SealedError(f"{d} already exists: the test is opened once (a failed attempt is moved aside, §J8.4)")
    return frozen


def authorize(**kw) -> str:
    global _AUTHORIZED
    _AUTHORIZED = check(**kw)
    return _AUTHORIZED


def _fsync(p: Path) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(staging: Path, out: Path) -> dict:
    """Hash every staged file into staging/output_manifest.json, fsync files and directories, then rename staging to
    out in one step. Returns the manifest."""
    files = sorted(p for p in staging.rglob("*") if p.is_file() and p != staging / "output_manifest.json")
    man = {"sha256": {p.relative_to(staging).as_posix(): sha256(p) for p in files}}
    (staging / "output_manifest.json").write_text(json.dumps(man, indent=1))
    for p in files + [staging / "output_manifest.json"]:
        _fsync(p)
    for d in sorted({p.parent for p in files} | {staging}, key=lambda q: -len(q.parts)):
        _fsync(d)
    if out.exists():
        raise SealedError(f"{out} appeared during the run")
    os.rename(staging, out)
    _fsync(out.parent)
    return man
