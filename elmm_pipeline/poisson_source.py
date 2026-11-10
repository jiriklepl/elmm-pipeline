"""One explicit local Poisson source snapshot for every GPU build path."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .build_receipts import file_identity

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARTS = ("CMakeLists.txt", "dependencies.lock.json", "src", "include", "cmake", "docs")


def dependency_lock():
    return json.loads((ROOT / "pressure-dependencies.json").read_text())


def _files(root):
    files = []
    for name in SOURCE_PARTS:
        path = root / name
        if path.is_symlink():
            raise ValueError("Poisson source symlinks are not supported: " + str(path))
        for entry in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if entry.is_symlink():
                raise ValueError("Poisson source symlinks are not supported: " + str(entry))
            if entry.is_file():
                files.append(entry)
    for name in ("CMakeLists.txt", "dependencies.lock.json", "src/device_api.cpp",
                 "include/poisson-solver/device_api.h"):
        if root / name not in files:
            raise ValueError("Incomplete local Poisson source: " + name)
    lock = json.loads((root / "dependencies.lock.json").read_text())
    if lock["dependencies"]["VkFFT"]["commit"] != dependency_lock()["VkFFT"]["revision"]:
        raise ValueError("Local Poisson and pipeline VkFFT pins differ")
    return files


def snapshot_solver(source, target, receipt):
    """Include local edits, exclude build products, and reject changed resumptions."""
    source = Path(source).resolve(strict=True)
    target = Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Poisson snapshot must be outside the supplied source tree")
    pin = dependency_lock()["provider"]["upstream_revision"]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != pin:
        ancestry = subprocess.run(
            ["git", "-C", str(source), "merge-base", "--is-ancestor", pin, revision],
            capture_output=True, text=True, check=False,
        )
        if ancestry.returncode != 0:
            raise ValueError("The local Poisson implementation must be based on " + pin)
    original = [file_identity(p) for p in _files(source)]
    receipt = Path(receipt)
    if receipt.exists():
        previous = verify_solver_snapshot(receipt)
        if previous["snapshot_root"] != str(target) or [r["original"] for r in previous["files"]] != original:
            raise ValueError("Local solver changed; prepare a fresh build")
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for before in original:
        path = Path(before["path"])
        dest = target / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            raise ValueError("Poisson snapshot contains a symlink: " + str(dest))
        if dest.exists():
            if file_identity(dest)["sha256"] != before["sha256"]:
                raise ValueError("Local solver changed; prepare a fresh build")
        else:
            shutil.copyfile(path, dest)
        if file_identity(path) != before or file_identity(dest)["sha256"] != before["sha256"]:
            raise ValueError("Local solver source changed during snapshot")
        records.append({"original": before, "snapshot": file_identity(dest)})
    if [file_identity(p) for p in _files(source)] != original:
        raise ValueError("Local solver source inventory changed during snapshot")
    result = {"base_revision": pin, "snapshot_root": str(target), "files": records}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    verify_solver_snapshot(receipt)
    return result


def verify_solver_snapshot(receipt):
    """Validate the complete immutable snapshot, without consulting a live checkout."""
    value = json.loads(Path(receipt).read_text())
    if (set(value) != {"base_revision", "snapshot_root", "files"}
            or value["base_revision"] != dependency_lock()["provider"]["upstream_revision"]):
        raise ValueError("Invalid local Poisson snapshot receipt")
    root = Path(value["snapshot_root"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("Poisson snapshot requires an absolute physical directory")
    expected = []
    for record in value["files"]:
        if set(record) != {"original", "snapshot"} or record["original"]["sha256"] != record["snapshot"]["sha256"]:
            raise ValueError("Poisson source snapshot does not match its recorded inputs")
        expected.append(record["snapshot"])
    actual = [file_identity(p) for p in _files(root)]
    if actual != expected:
        raise ValueError("Poisson source snapshot changed")
    # Detect stale or injected files outside the supported source inventory too.
    all_paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in all_paths) or {str(p) for p in all_paths if p.is_file()} != {r["path"] for r in actual}:
        raise ValueError("Poisson snapshot contains obsolete or additional files; prepare a fresh build")
    return value
