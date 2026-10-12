"""Prepare, execute and seal explicit, independently versioned campaigns."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .identity import write
from .launch_environment import apply_launch_environment, launch_environment_values
from .prepare import check, prepare, validate_cpu_placement


def external(root, phase):
    state = check(root)
    spec = state["commands"][phase]
    def expand(value):
        return value.replace("{campaign}", str(root)).replace("{python}", sys.executable)
    argv = [expand(value) for value in spec["argv"]]
    overrides = {key: expand(value) for key, value in spec.get("environment", {}).items()}
    env = {**os.environ, **overrides}
    placement = state.get("cpu_placement")
    validate_cpu_placement(placement)
    placement_environment = {}
    if placement is not None:
        placement_environment = {"OMP_NUM_THREADS": str(state["threads"]),
                                 "OMP_DYNAMIC": "FALSE", **placement["environment"]}
        if (any(key in overrides for key in ("OMP_PLACES", "GOMP_CPU_AFFINITY"))
                or any(key in overrides and overrides[key].lower() != value.lower()
                       for key, value in placement_environment.items())):
            raise ValueError("external phase environment conflicts with calibrated CPU placement")
        env.pop("OMP_PLACES", None)
        env.pop("GOMP_CPU_AFFINITY", None)
        env.update(placement_environment)
    apply_launch_environment(env, state.get("openmp_launch_environment"), overrides=overrides)
    env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
    write(root / (phase + "-command.json"), {"argv": argv, "cwd": expand(spec.get("cwd", "{campaign}")),
          "explicit_environment": spec.get("environment", {}),
          "openmp_launch_environment": launch_environment_values(env),
          "placement_environment": placement_environment, "timings_reused": False})
    subprocess.run(argv, cwd=expand(spec.get("cwd", "{campaign}")), env=env, check=True)
    check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    snapshot_parser = sub.add_parser("plot-snapshot")
    snapshot_parser.add_argument("--snapshot", type=Path, required=True)
    snapshot_parser.add_argument("--output", type=Path, required=True)
    snapshot_parser.add_argument("--native-label", default="Native")
    snapshot_parser.add_argument("--gpu-label", default="Scoped GPU")
    for name in ("run", "worker", "plot", "seal"):
        child = sub.add_parser(name)
        child.add_argument("--campaign", type=Path, required=True)
        if name in ("run", "worker"):
            child.add_argument("phase")
        if name in ("plot", "seal"):
            child.add_argument("--output", type=Path, required=name == "seal")
        if name == "plot":
            child.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.config, args.output))
        return
    if args.command == "plot-snapshot":
        from .report import plot_snapshot
        print(plot_snapshot(args.snapshot, args.output, gpu_label=args.gpu_label, native_label=args.native_label))
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
        plot(root, args.output, historical=args.historical)
    elif args.command == "seal":
        from .report import seal
        print(seal(root, args.output))


if __name__ == "__main__":
    main()
