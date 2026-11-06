"""Atomic file outputs: write to a temporary sibling, then os.replace, so a reader never sees a partial file."""
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path):
    """yield a temporary sibling path; on success it atomically replaces `path`."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text(path, s: str):
    with atomic_path(path) as t:
        t.write_text(s)
