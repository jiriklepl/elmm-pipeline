"""Bounded CUDA/NVTX diagnostics of a frozen application, outside timing samples.

Invoke under ``elmm_pipeline.campaigns.guard``. The application continues after
collection ends so its complete output can still be validated. This module does
not extrapolate a short device trace into a complete-application cost estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

from ..gpu_profiling import parse_nsys_sqlite
from ..validation import compare_vtk, read_vtk, validate_log
from .identity import digest, write
from .launch_environment import launch_environment_values
from .prepare import check
from .run import environment
from .pressure_cache import apply_pressure_cache


# The profiler may return when collection ends while its target keeps running.
# This stdlib-only target owns the application log and atomically publishes its
# actual exit status after subprocess.run has reaped the application.
TARGET_WRAPPER = '''from pathlib import Path
import json
import os
import subprocess
import sys

completion, logfile, token = sys.argv[1:4]
status = {"schema_version": 1, "token": token}
with open(logfile, "w") as log:
    try:
        status["exit_code"] = subprocess.run(sys.argv[4:], stdout=log,
            stderr=subprocess.STDOUT, check=False).returncode
    except OSError as error:
        status.update(exit_code=127, launch_error=str(error))
temporary = Path(completion + ".tmp")
temporary.write_text(json.dumps(status) + "\\n")
os.replace(temporary, completion)
code = status["exit_code"]
sys.exit(code if code >= 0 else 128 - code)
'''


def wait_completion(path, token, deadline):
    """Confirm this target exited within the same total profiling timeout."""
    while True:
        if path.is_file():
            try:
                status = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise RuntimeError("invalid profiled application completion marker") from error
            if (not isinstance(status, dict) or status.get("schema_version") != 1 or status.get("token") != token
                    or type(status.get("exit_code")) is not int):
                raise RuntimeError("invalid profiled application completion marker")
            return status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("profiled application did not publish completion before the timeout")
        time.sleep(min(0.05, remaining))


def command(profiler, prefix, binary, arguments, seconds):
    if type(seconds) is not int or not 1 <= seconds <= 120:
        raise ValueError("capture duration must be between one and 120 seconds")
    return [str(profiler), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
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
    target, completion, application_log = (output / name for name in
        ("profiler-target.py", "application-completion.json", "application.log"))
    token = secrets.token_hex(16)
    application_argv = [str(root / selected_mode["binary"]),
                        *selected_case["configuration"]["runtime_arguments"]]
    argv = command(executable, prefix, sys.executable,
                   [str(target), str(completion), str(application_log), token, *application_argv], seconds)
    env = environment(state["threads"], traced=False, cpu_placement=state.get("cpu_placement"),
                      openmp_launch_environment=state.get("openmp_launch_environment"))
    env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
    cache = apply_pressure_cache(root, state, selected_mode, env)
    shutil.copytree(root / selected_case["config"], output)
    if any(path.exists() for path in (target, completion, application_log)):
        raise ValueError("configuration collides with profiler target completion files")
    target.write_text(TARGET_WRAPPER)
    record = {"schema_version": 1, "status": "in_progress", "case": case, "mode": mode,
              "argv": argv, "cwd": str(output), "campaign_sha256": digest(root / "campaign.json"),
              "profiler_sha256": digest(executable), "helper_sha256": digest(__file__),
              "target_wrapper_sha256": digest(target), "target_python_sha256": digest(sys.executable),
              "application_argv": application_argv, "application_log": str(application_log),
              "completion_marker": str(completion), "application_completed": False,
              "binary_sha256": digest(root / selected_mode["binary"]), "maximum_capture_seconds": seconds,
              "phase_timer_enabled": False, "runtime_trace_enabled": False, "timing_sample": False,
              "threads": state["threads"], "gpu_uuid": state["selected_gpu_uuid"],
              "openmp_launch_environment": launch_environment_values(env),
              "pressure_kernel_cache": cache,
              "coverage": "bounded trace; full-run device totals require independent count/byte agreement",
              "overlap_claimed": False, "durations_are_additive_application_budget": False}
    write(output / "report.json", record)
    started = time.perf_counter()
    deadline = time.monotonic() + state.get("run_timeout_seconds", 900)
    try:
        with (output / "profiler.log").open("w") as log:
            result = subprocess.run(argv, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    check=False, timeout=max(0, deadline - time.monotonic()))
        record.update(process_exit_code=result.returncode, profiler_exit_code=result.returncode,
                      profiler_wall_seconds=time.perf_counter() - started)
        if result.returncode and not completion.is_file():
            raise RuntimeError("bounded profiler failed; preserve " + str(output / "profiler.log"))
        waiting = time.perf_counter()
        status = wait_completion(completion, token, deadline)
        record.update(application_completed=True, application_exit_code=status["exit_code"],
                      completion_wait_seconds=time.perf_counter() - waiting,
                      profiled_execution_wall_seconds=time.perf_counter() - started)
        if status["exit_code"]:
            raise RuntimeError("profiled application failed with exit code " + str(status["exit_code"]))
        if result.returncode:
            raise RuntimeError("bounded profiler failed; preserve " + str(output / "profiler.log"))
        log = application_log.read_text(errors="replace")
        record["timing"] = validate_log(log, selected_case["configuration"]["expected_max_steps"],
                                        expected_completed_steps=selected_case["completed_steps"])
        record["validation"] = compare_vtk(read_vtk(root / selected_case["reference"]), read_vtk(output / "output/out.vtk"))
        record["metrics"] = parse_nsys_sqlite(prefix.with_suffix(".sqlite"))
    except Exception as error:
        record.update(status="failed", failure_reason=str(error))
        write(output / "report.json", record)
        raise
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
