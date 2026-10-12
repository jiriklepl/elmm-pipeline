import json
from pathlib import Path
import shutil
import sys
import venv

from .common import ROOT, command, digest_tree, python_environment, write_json
from .build import bootstrap_native


def ensure_repo(name, spec):
    """Reuse a clean cache, fetching an explicitly changed commit pin if needed."""
    destination = ROOT / ".cache" / "repos" / name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--no-checkout", spec["url"], destination])
        command(["git", "checkout", "--detach", "--no-overwrite-ignore", spec["revision"]], cwd=destination)
    dirty = command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=destination)[0]
    if dirty:
        raise RuntimeError(f"Cached {name} checkout has source changes; preserve it and refresh the cache")
    revision = command(["git", "rev-parse", "HEAD"], cwd=destination)[0].strip()
    if revision != spec["revision"]:
        try:
            command(["git", "cat-file", "-e", spec["revision"] + "^{commit}"], cwd=destination)
        except RuntimeError:
            command(["git", "fetch", "--no-tags", spec["url"], spec["revision"]], cwd=destination)
        # No reset/clean/force: even a concurrent worktree edit must be preserved.
        command(["git", "checkout", "--detach", "--no-overwrite-ignore", spec["revision"]], cwd=destination)
        revision = command(["git", "rev-parse", "HEAD"], cwd=destination)[0].strip()
        if revision != spec["revision"]:
            raise RuntimeError(f"{destination} is at {revision}, expected pinned {spec['revision']}")
    return destination


def ensure_environment(tool):
    if tool != "local":
        raise ValueError(f"Unknown transformation tool: {tool}; expected local")
    destination = ROOT / ".cache" / "envs" / tool
    python = destination / "bin" / "python"
    requirements = ROOT / f"requirements-{tool}.txt"
    marker = destination / "requirements.installed"
    expected = requirements.read_text()
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(destination)
    if not marker.exists() or marker.read_text() != expected:
        command([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", requirements],
                env=python_environment(), log=ROOT / ".cache" / f"install-{tool}.log")
        marker.write_text(expected)
    return python


def setup(tools, local_source=None):
    if any(tool not in {"baseline", "local"} for tool in tools):
        raise ValueError("Setup tools must be baseline or local")
    pins = json.loads((ROOT / "dependencies.json").read_text())
    for name, spec in pins.items():
        print(f"Fetching/verifying {name}", flush=True)
        ensure_repo(name, spec)
    print("Checking native build dependencies", flush=True)
    bootstrap_native(ROOT)
    if "local" in tools:
        destination = ROOT / ".cache" / "repos" / "local-compiler"
        if local_source:
            source = Path(local_source).resolve()
            if not (source / "compiler" / "__main__.py").is_file():
                raise ValueError("--local-compiler must contain compiler/__main__.py")
            if source == destination or destination in source.parents:
                raise ValueError("Supply an external compiler checkout, not its cache snapshot")
            provenance = {"source": str(source), "compiler_sha256": digest_tree(source / "compiler")}
            try:
                provenance["git_revision"] = command(["git", "rev-parse", "HEAD"], cwd=source)[0].strip()
                provenance["git_status"] = command(["git", "status", "--short", "--", "compiler"], cwd=source)[0]
            except RuntimeError:
                pass
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            shutil.copytree(source / "compiler", destination / "compiler",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            write_json(destination / "source.json", provenance)
        elif not (destination / "compiler" / "__main__.py").is_file():
            raise ValueError("First setup with local requires --local-compiler /explicit/checkout/path")
    packages = {}
    for tool in tools:
        if tool == "baseline":
            continue
        print(f"Preparing {tool} Python environment", flush=True)
        python = ensure_environment(tool)
        packages[tool] = command([python, "-m", "pip", "freeze"], env=python_environment())[0].splitlines()
    write_json(ROOT / ".cache" / "setup.json", {
        "pins": pins, "tools": tools, "python": sys.version, "packages": packages,
        "native": json.loads((ROOT / ".cache/native/libraries.json").read_text()),
    })
    print(f"Setup complete: {ROOT}", flush=True)
