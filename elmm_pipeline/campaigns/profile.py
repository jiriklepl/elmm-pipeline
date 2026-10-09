"""Bounded CUDA/NVTX diagnostics of a frozen application, outside timing samples.

Invoke under ``elmm_pipeline.campaigns.guard``. The application continues after
collection ends so its complete output can still be validated. This module does
not extrapolate a short device trace into a complete-application cost estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import time

from ..gpu_profiling import parse_nsys_sqlite
from ..validation import compare_vtk, read_vtk, validate_log
from .identity import digest, write
from .prepare import check
from .run import environment


def command(profiler, prefix, binary, arguments, seconds):
    if type(seconds) is not int or not 1 <= seconds <= 30:
        raise ValueError("capture duration must be between one and thirty seconds")
    return [str(profiler), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--backtrace=none", "--cudabacktrace=none", "--resolve-symbols=false",
            "--cuda-memory-usage=false", "--cuda-event-trace=false", "--cuda-trace-all-apis=false",
            "--cuda-flush-interval=0", "--duration=" + str(seconds), "--kill=none", "--export=sqlite",
            "--force-overwrite=false", "--output=" + str(prefix), str(binary), *arguments]


def profile(root, output, *, case, mode, profiler, seconds=2):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    state = check(root)
    selected_case = next(row for row in state["cases"] if row["name"] == case)
    selected_mode = next(row for row in state["modes"] if row["name"] == mode)
    executable = shutil.which(str(profiler))
    if not executable:
        raise ValueError("explicit Nsight Systems executable was not found")
    prefix = output / "capture"
    argv = command(executable, prefix, root / selected_mode["binary"],
                   selected_case["configuration"]["runtime_arguments"], seconds)
    shutil.copytree(root / selected_case["config"], output)
    env = environment(state["threads"], traced=False)
    env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
    record = {"schema_version": 1, "status": "in_progress", "case": case, "mode": mode,
              "argv": argv, "cwd": str(output), "campaign_sha256": digest(root / "campaign.json"),
              "profiler_sha256": digest(executable), "helper_sha256": digest(__file__),
              "binary_sha256": digest(root / selected_mode["binary"]), "maximum_capture_seconds": seconds,
              "phase_timer_enabled": False, "runtime_trace_enabled": False, "timing_sample": False,
              "threads": state["threads"], "gpu_uuid": state["selected_gpu_uuid"],
              "coverage": "bounded trace; full-run device totals require independent count/byte agreement",
              "overlap_claimed": False, "durations_are_additive_application_budget": False}
    write(output / "report.json", record)
    started = time.perf_counter()
    with (output / "profiler.log").open("w") as log:
        result = subprocess.run(argv, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                                check=False, timeout=state.get("run_timeout_seconds", 900))
    record.update(process_exit_code=result.returncode, profiler_wall_seconds=time.perf_counter() - started)
    write(output / "report.json", record)
    if result.returncode:
        raise RuntimeError("bounded profiler failed; preserve " + str(output / "profiler.log"))
    log = (output / "profiler.log").read_text(errors="replace")
    record["timing"] = validate_log(log, selected_case["configuration"]["expected_max_steps"],
                                    expected_completed_steps=selected_case["completed_steps"])
    record["validation"] = compare_vtk(read_vtk(root / selected_case["reference"]), read_vtk(output / "output/out.vtk"))
    record["metrics"] = parse_nsys_sqlite(prefix.with_suffix(".sqlite"))
    record["status"] = "passed" if record["validation"]["passed"] else "failed"
    write(output / "report.json", record)
    check(root)
    if record["status"] != "passed":
        raise RuntimeError("profiled complete application fields disagree")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--nsys", required=True)
    parser.add_argument("--seconds", type=int, default=2)
    args = parser.parse_args()
    profile(args.campaign, args.output, case=args.case, mode=args.mode, profiler=args.nsys, seconds=args.seconds)


if __name__ == "__main__":
    main()
