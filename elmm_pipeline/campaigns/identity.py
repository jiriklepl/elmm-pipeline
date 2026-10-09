"""Content identities for committed inputs and immutable campaign receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tree_hashes(path):
    root = Path(path)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not EXCLUDED.intersection(p.relative_to(root).parts)
            and p.suffix != ".pyc"}


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def repository(root, paths):
    """Only the declared committed inputs enter a snapshot, never the worktree."""
    root = Path(root).resolve(strict=True)
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    names = git(root, "ls-tree", "-r", "--name-only", commit, "--", *paths).decode().splitlines()
    if not names:
        raise ValueError("repository input selection is empty")
    hashes = {}
    for name in names:
        path = root / name
        data = git(root, "show", commit + ":" + name)
        expected = hashlib.sha256(data).hexdigest()
        if not path.is_file() or digest(path) != expected:
            raise ValueError("selected worktree differs from its commit: " + str(path))
        hashes[name] = expected
    return {"root": str(root), "commit": commit, "paths": paths, "sha256": hashes}


def check_files(root, hashes):
    for name, expected in hashes.items():
        if digest(Path(root) / name) != expected:
            raise ValueError("frozen artifact changed: " + str(Path(root) / name))


def guard_passed(path):
    value = read(path)
    if (value.get("status") != "finished" or value.get("exit_code") != 0
            or value.get("stopped_for_memory") or value.get("interrupted_signal")
            or value.get("headroom_bytes") != 3 * 2**30
            or value.get("minimum_available_bytes", 0) < 3 * 2**30
            or value.get("cleanup_complete") is not True or value.get("remaining_descendants")):
        raise ValueError("successful memory guard and descendant cleanup required: " + str(path))
    return value
