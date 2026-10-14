"""Fresh application runs; diagnostics never enter the wall-time sample set."""
from __future__ import annotations

import math
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from ..common import clean_environment
from ..gpu_validation import gpu_activity, trace_environment
from ..validation import compare_vtk, read_vtk, validate_log
from .batches import batch_summary
from .guard import available
from .identity import digest, read, write
from .launch_environment import apply_launch_environment, launch_environment_values
from .prepare import check, validate_cpu_placement
from .targets import evaluate_target, target_seconds
from .transfers import aggregate_final_statistics


def planning_diagnostics(log):
    totals = {phase: {"seconds": 0.0, "calls": 0, "cache_hits": 0}
              for phase in ("query_construction", "validation", "selection")}
    pattern = (r"(?m)^FORT_SCOPED planning_timing context=\d+ phase=(query_construction|validation|selection) "
               r"seconds=(\S+) cache_hit=([01]) query_generation=\d+ state_generation=\d+$")
    for phase, encoded, cached in re.findall(pattern, log):
        seconds = float(encoded)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("public planning duration must be finite and nonnegative")
        totals[phase]["seconds"] += seconds
        totals[phase]["calls"] += 1
        totals[phase]["cache_hits"] += int(cached)
    return {"phases": totals, "timing_sample": False, "trace_overhead_present": True}


def environment(threads, *, traced, cpu_placement=None, openmp_launch_environment=None):
    validate_cpu_placement(cpu_placement)
    env = trace_environment(clean_environment(threads), "local", enabled=traced)
    for name in list(env):
        if name.startswith("FORT_") and name not in {"FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE"}:
            env.pop(name)
    if cpu_placement is not None:
        env.pop("OMP_PLACES", None)
        env.pop("GOMP_CPU_AFFINITY", None)
        env.update(cpu_placement["environment"])
    apply_launch_environment(env, openmp_launch_environment)
    return env


def run_one(root, state, case, mode, directory, *, traced):
    if available() < 3 * 2**30:
        raise RuntimeError("less than 3 GiB available; no workload started")
    env = environment(state["threads"], traced=traced and mode["name"] != "native",
                      cpu_placement=state.get("cpu_placement"),
                      openmp_launch_environment=state.get("openmp_launch_environment"))
    if directory.exists():
        raise ValueError("existing run evidence is immutable: " + str(directory))
    shutil.copytree(root / case["config"], directory)
    argv = [str(root / mode["binary"]), *case["configuration"]["runtime_arguments"]]
    rss = directory / "peak-rss-kib.txt"
    command = [state["tools"]["time"]["path"], "--format=%M", "--output=" + str(rss), *argv]
    env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
    print("TRACE" if traced else "TIMING", case["name"], mode["name"], directory.name, flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True,
                            check=False, timeout=state.get("run_timeout_seconds", 900))
    wall = time.perf_counter() - started
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("application failed: " + str(directory / "stderr.log"))
    timing = validate_log(result.stdout, case["configuration"]["expected_max_steps"],
                          expected_completed_steps=case["completed_steps"])
    validation = compare_vtk(read_vtk(root / case["reference"]), read_vtk(directory / "output/out.vtk"))
    record = {"mode": mode["name"], "argv": argv, "process_measurement_argv": command,
              "directory": str(directory), "timing": timing, "validation": validation,
              "peak_host_rss_bytes": int(rss.read_text().strip()) * 1024, "traced": traced,
              "environment": {key: env.get(key) for key in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PLACES",
                  "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "CUDA_VISIBLE_DEVICES")},
              "openmp_launch_environment": launch_environment_values(env),
              "diagnostic_wall_seconds" if traced else "wall_seconds": wall}
    if traced:
        adapter = read(root / mode["adapter"]) if "adapter" in mode else {}
        record["activity"] = gpu_activity(result.stdout + result.stderr, "local", adapter.get("regions", []),
                                           scope_manifest=adapter.get("scope_manifest"),
                                           scope_packages=adapter.get("scope_packages"))
        record["planning_diagnostics"] = planning_diagnostics(result.stderr)
        events = record["activity"].get("scoped_runtime", {}).get("modeled_evidence", [])
        if any(row["status"] != 0 for row in events if row["event"] == "definition_validation"):
            raise RuntimeError("ordered source definition validation failed")
        record["transfer_statistics"] = aggregate_final_statistics(events)
        record["batch_statistics"] = batch_summary(events)
    write(directory / "run.json", record)
    if not validation["passed"]:
        raise RuntimeError("complete field comparison failed: " + str(directory / "run.json"))
    return record


def summaries(runs, modes):
    summary = {}
    for mode in modes:
        samples = [row for row in runs if row["mode"] == mode["name"] and not row.get("warmup")]
        if any(row["traced"] for row in samples):
            raise ValueError("diagnostics cannot be wall-time samples")
        values = [row["wall_seconds"] for row in samples]
        summary[mode["name"]] = {"median": statistics.median(values), "min": min(values), "max": max(values),
                                "repetitions": len(values), "samples": values,
                                "peak_host_rss_bytes_max": max(row["peak_host_rss_bytes"] for row in samples)}
    for row in summary.values():
        row["ratio_to_native"] = row["median"] / summary["native"]["median"]
    return summary


def gates(state, summary, case=None, runs=()):
    if case and "performance_target" in case:
        return evaluate_target(state, case, summary, runs)
    baseline = min(state["baseline_modes"], key=lambda mode: summary[mode]["median"])
    record = {"baseline_mode": baseline, "baseline_seconds": summary[baseline]["median"],
              "target_seconds": 0.70 * summary[baseline]["median"]}
    candidate = state.get("candidate_mode")
    if candidate:
        ratio = summary[candidate]["median"] / summary[baseline]["median"]
        record.update(candidate_mode=candidate, candidate_ratio=ratio,
                      thirty_percent_lower=ratio <= 0.70, within_five_percent=ratio <= 1.05)
    return record


def borderline(state, summary, case=None):
    if case and "performance_target" in case:
        target = case["performance_target"]
        candidate = summary[state["candidate_mode"]]["median"]
        return any(abs(candidate / reference - 1) <= .03 for reference in (
            target_seconds(target, summary), summary[target["control_mode"]]["median"]))
    reference = min(summary[mode]["median"] for mode in state["baseline_modes"])
    return any(abs(summary[mode["name"]]["median"] / reference - gate) <= 0.03
               for mode in state["modes"] if mode["name"] not in state["baseline_modes"]
               for gate in (0.70, 1.0, 1.05))


def gpu_identity(root, state, phase):
    command = [state["tools"]["nvidia-smi"]["path"], "--query-gpu=uuid,name,driver_version,compute_cap",
               "--format=csv,noheader"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    profile = read(root / state["profile"])
    record = {"argv": command, "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
              "expected_uuid": profile["hardware"]["gpu_uuid"], "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}
    write(root / (phase + "-gpu.json"), record)
    if result.returncode or profile["hardware"]["gpu_uuid"] not in result.stdout:
        raise RuntimeError("calibrated GPU unavailable")


def run(root, phase):
    root = Path(root).resolve(strict=True)
    state = check(root)
    if sorted(os.sched_getaffinity(0)) != state["host"]["affinity"]:
        raise ValueError("host CPU placement changed since preparation")
    gpu_identity(root, state, phase)
    output = root / phase
    output.mkdir(exist_ok=False)
    traced = phase == "verification"
    if not traced:
        verified = read(root / "verification/report.json")
        if verified["status"] != "passed" or verified["campaign_sha256"] != digest(root / "campaign.json"):
            raise ValueError("exact campaign must pass verification before timing")
        if read(root / "halos/report.json")["status"] != "passed":
            raise ValueError("complete array and halo verification must pass before timing")
    report = {"schema_version": 1, "status": "in_progress", "campaign_sha256": digest(root / "campaign.json"),
              "threads": state["threads"], "cases": [], "profile_sha256": state["profile_sha256"],
              "runtime_id": state["runtime_id"], "traced": traced, "timings_reused": False,
              "policy_promoted": False, "phase_timer_enabled": False,
              "wall_time_includes": "process startup, computation, native solver, transfers, output and teardown"}
    write(output / "report.json", report)
    modes = state["modes"]
    for case in state["cases"]:
        entry = {"case": case["name"], "runs": []}
        report["cases"].append(entry)
        repetitions = 0 if traced else state.get("repetitions", 3)
        round_ = 0
        while round_ <= repetitions:
            order = modes[round_ % len(modes):] + modes[:round_ % len(modes)]
            for mode in order:
                run = run_one(root, state, case, mode, output / case["name"] / mode["name"] / f"round-{round_:02}",
                              traced=traced)
                run.update(warmup=not traced and round_ == 0, round=round_)
                entry["runs"].append(run)
                write(output / "report.json", report)
            round_ += 1
            if not traced and round_ == 4 and borderline(state, summaries(entry["runs"], modes), case):
                repetitions = 7
                entry["extended_reason"] = "within three percentage points of tie or acceptance boundary"
        if not traced:
            entry["summary"] = summaries(entry["runs"], modes)
            entry["gates"] = gates(state, entry["summary"], case, entry["runs"])
        write(output / "report.json", report)
    check(root)
    report["status"] = "passed"
    write(output / "report.json", report)
