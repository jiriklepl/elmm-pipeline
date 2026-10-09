"""Prepare, execute and seal explicit, independently versioned campaigns."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .identity import read, write
from .prepare import check, prepare


def external(root, phase):
    state = check(root)
    spec = state["commands"][phase]
    def expand(value):
        return value.replace("{campaign}", str(root)).replace("{python}", sys.executable)
    argv = [expand(value) for value in spec["argv"]]
    env = {**os.environ, **{key: expand(value) for key, value in spec.get("environment", {}).items()}}
    env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
    write(root / (phase + "-command.json"), {"argv": argv, "cwd": expand(spec.get("cwd", "{campaign}")),
          "explicit_environment": spec.get("environment", {}), "timings_reused": False})
    subprocess.run(argv, cwd=expand(spec.get("cwd", "{campaign}")), env=env, check=True)
    check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    for name in ("run", "worker", "plot", "seal"):
        child = sub.add_parser(name)
        child.add_argument("--campaign", type=Path, required=True)
        if name in ("run", "worker"):
            child.add_argument("phase")
        if name == "seal":
            child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.config, args.output))
        return
    root = args.campaign.resolve(strict=True)
    state = check(root)
    if args.command == "run":
        if args.phase not in {"verification", "timings", "halos", *state.get("commands", {})}:
            parser.error("unknown configured phase")
        # Execute the committed runner included in the receipt, independently
        # of whichever checkout submitted the command.
        env = {**os.environ, "PYTHONPATH": str(root / "sources/pipeline"), "PYTHONDONTWRITEBYTECODE": "1"}
        command = [sys.executable, "-m", "elmm_pipeline.campaigns.guard",
                   "--record", str(root / "guards" / (args.phase + ".json")),
                   "--log", str(root / "logs" / (args.phase + ".log")), "--",
                   sys.executable, "-m", "elmm_pipeline.campaigns", "worker", "--campaign", str(root), args.phase]
        raise SystemExit(subprocess.run(command, cwd=root, env=env, check=False).returncode)
    if args.command == "worker":
        if args.phase in ("verification", "timings"):
            from .run import run
            run(root, args.phase)
        elif args.phase == "halos":
            from .halos import run
            run(root)
        else:
            external(root, args.phase)
    elif args.command == "plot":
        from .report import plot
        plot(root)
    elif args.command == "seal":
        from .report import seal
        print(seal(root, args.output))


if __name__ == "__main__":
    main()
