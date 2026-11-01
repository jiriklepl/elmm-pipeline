"""Collect GPU execution evidence separately from application timing samples.

Nsight Systems CUDA activity timestamps measure device kernel and copy work.
Duration totals are sums of recorded CUDA activities, not elapsed
critical-path time when work overlaps. Profiling wall time includes tracing
and export overhead.
"""

from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
import uuid


def _empty_metrics(reason: str) -> dict:
    return {"available": False, "status": "unavailable", "reason": reason,
            "kernel_seconds": None, "kernel_count": None,
            "graph_aggregate_seconds": None, "graph_aggregate_count": None,
            "graph_node_kernel_count": None, "graph_timing_available": False,
            "kernel_inventory_complete": None,
            "kernel_inventory_reason": "No independent full-application kernel coverage check was performed",
            "transfer_seconds": None, "transfer_count": None,
            "h2d_bytes": None, "d2h_bytes": None, "transfer_bytes": None,
            "gpu_span_seconds": None, "kernels": [], "transfers": [],
            "kernel_timing_available": False, "transfer_timing_available": False,
            "measurement": "sum of device activity durations in a separate profiled run"}


def parse_nsys_sqlite(path: Path) -> dict:
    """Aggregate actual CUDA kernel/copy events from an Nsight SQLite export.

    Missing activity tables or empty tables stay unavailable (None), avoiding
    an apparent zero-cost GPU result when CUPTI tracing was unsupported. Kernel
    and transfer availability are recorded separately for compute-only traces.
    Graph aggregates remain separate and mark the kernel inventory incomplete;
    node activity alone does not prove that the whole application was captured.
    """
    path = path.resolve()
    metrics = _empty_metrics("No CUDA kernel activity was recorded")
    metrics["sqlite_path"] = str(path)
    if not path.is_file():
        metrics["reason"] = "Nsight Systems did not produce a SQLite export"
        return metrics
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            metrics["activity_tables"] = sorted(name for name in tables if name.startswith("CUPTI_ACTIVITY_KIND_"))
            starts, ends = [], []
            # CONCURRENT_KERNEL is used by some older CUPTI exports. Select one
            # table rather than adding both representations of the same work.
            kernel_table = next((name for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if name in tables), None)
            if kernel_table:
                count, nanoseconds, first, last = connection.execute(
                    f'SELECT COUNT(*), SUM("end"-start), MIN(start), MAX("end") FROM {kernel_table}'
                ).fetchone()
                if count:
                    metrics.update(available=True, status="ok", reason=None,
                                   kernel_count=count, kernel_seconds=nanoseconds / 1e9)
                    starts.append(first)
                    ends.append(last)
                    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({kernel_table})")}
                    if "graphNodeId" in columns:
                        # CUPTI uses zero for non-graph launches; Nsight
                        # exports may instead represent that absence as NULL.
                        metrics["graph_node_kernel_count"] = connection.execute(
                            f'SELECT COUNT(*) FROM {kernel_table} '
                            'WHERE graphNodeId IS NOT NULL AND graphNodeId != 0'
                        ).fetchone()[0]
                    name_column = next((name for name in ("demangledName", "shortName") if name in columns), None)
                    if name_column and "StringIds" in tables:
                        rows = connection.execute(
                            f'SELECT s.value, COUNT(*), SUM(k."end"-k.start) FROM {kernel_table} k '
                            f'LEFT JOIN StringIds s ON s.id=k.{name_column} GROUP BY k.{name_column} '
                            'ORDER BY SUM(k."end"-k.start) DESC'
                        )
                        metrics["kernels"] = [{"name": name, "count": count, "seconds": duration / 1e9}
                                              for name, count, duration in rows]
            if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
                count, nanoseconds, first, last = connection.execute(
                    'SELECT COUNT(*), SUM("end"-start), MIN(start), MAX("end") '
                    'FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE'
                ).fetchone()
                if count:
                    # Default graph tracing records aggregate intervals instead
                    # of constituent kernels. These may overlap node/copy events;
                    # never add their durations or counts to kernel totals.
                    metrics.update(available=True, status="partial",
                                   reason="CUDA graph aggregates omit individual graph kernel activity",
                                   graph_aggregate_count=count, graph_aggregate_seconds=nanoseconds / 1e9,
                                   graph_timing_available=True, kernel_inventory_complete=False,
                                   kernel_inventory_reason="Graph aggregates prevent a complete individual-kernel inventory")
                    starts.append(first)
                    ends.append(last)
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
                rows = connection.execute(
                    'SELECT copyKind, COUNT(*), SUM("end"-start), SUM(bytes), MIN(start), MAX("end") '
                    'FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind'
                ).fetchall()
                labels = {}
                if "ENUM_CUDA_MEMCPY_OPER" in tables:
                    labels = {row[0]: (row[1], row[2]) for row in connection.execute("SELECT id,name,label FROM ENUM_CUDA_MEMCPY_OPER")}
                # These are CUPTI's documented enum identities, used only when
                # older exports omit the human-readable enumeration table.
                h2d_ids = {key for key, (name, _) in labels.items() if name == "CUDA_MEMCPY_KIND_HTOD"} if labels else {1}
                d2h_ids = {key for key, (name, _) in labels.items() if name == "CUDA_MEMCPY_KIND_DTOH"} if labels else {2}
                if rows:
                    metrics.update(transfer_count=sum(row[1] for row in rows),
                                   transfer_seconds=sum(row[2] for row in rows) / 1e9,
                                   transfer_bytes=sum(row[3] for row in rows),
                                   h2d_bytes=sum(row[3] for row in rows if row[0] in h2d_ids),
                                   d2h_bytes=sum(row[3] for row in rows if row[0] in d2h_ids))
                    for kind, count, duration, size, first, last in rows:
                        starts.append(first)
                        ends.append(last)
                        label = labels.get(kind, (None, {1: "Host-to-Device", 2: "Device-to-Host"}.get(kind, str(kind))))[1]
                        metrics["transfers"].append({"kind": label, "copy_kind": kind, "count": count,
                                                     "seconds": duration / 1e9, "bytes": size})
            if starts:
                metrics["gpu_span_seconds"] = (max(ends) - min(starts)) / 1e9
            metrics["kernel_timing_available"] = metrics["kernel_seconds"] is not None
            metrics["transfer_timing_available"] = metrics["transfer_seconds"] is not None
    except (sqlite3.Error, OSError, TypeError, ValueError) as error:
        metrics = _empty_metrics(f"Could not read Nsight CUDA activities: {error}")
        metrics["sqlite_path"] = str(path)
    return metrics


def profile_command(
    binary_args: list[str],
    directory: Path,
    env: dict[str, str],
    cwd: Path,
    timeout: float = 300,
    *,
    nsys: str | Path | None = None,
) -> tuple[str, float, dict]:
    """Profile one untimed application run and return log, wall time and metrics.

    CPU sampling and context-switch tracing are disabled to work without Linux
    performance-counter permissions, including WSL. If the profiler is missing,
    no application is launched and ``application_launched`` is false. Failures
    and timeouts retain logs and produce unavailable metrics instead of guessed
    timings. Callers keep ordinary benchmark samples entirely unprofiled.
    """
    if not binary_args or timeout <= 0:
        raise ValueError("A nonempty application command and positive profiling timeout are required")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    profiler = shutil.which(str(nsys or "nsys"))
    if profiler is None and nsys is None and Path("/usr/local/cuda/bin/nsys").is_file():
        profiler = "/usr/local/cuda/bin/nsys"
    if profiler is None:
        metrics = _empty_metrics("Nsight Systems was not found; install nsys or supply its path")
        metrics["application_launched"] = False
        metrics["application_executed"] = False
        (directory / "gpu-profile-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics["reason"] + "\n", 0.0, metrics
    prefix = directory / ("gpu-profile-" + uuid.uuid4().hex[:8])
    log_path = prefix.with_suffix(".log")
    command = [profiler, "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
               "--export=sqlite", "--force-overwrite=true", "--output=" + str(prefix), *map(str, binary_args)]
    (directory / "gpu-profile-command.json").write_text(
        json.dumps({"argv": command, "cwd": str(cwd.resolve()), "timeout_seconds": timeout,
                    "tracing_environment": {key: env[key] for key in ("FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE") if key in env}}, indent=2) + "\n",
        encoding="utf-8")
    started = time.perf_counter()
    error = None
    process = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                error = f"Nsight Systems profiling exceeded {timeout:g} seconds"
        except OSError as exception:
            error = f"Nsight Systems could not launch: {exception}"
    wall = time.perf_counter() - started
    combined_log = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_nsys_sqlite(prefix.with_suffix(".sqlite"))
    if error or (process and process.returncode):
        metrics.update(available=False, status="error", reason=error or f"Nsight Systems/application returned {process.returncode}")
    metrics.update(profiler=str(profiler), log_path=str(log_path), report_path=str(prefix.with_suffix(".nsys-rep")),
                   profiler_wall_seconds=wall, process_returncode=process.returncode if process else None,
                   profiler_started=process is not None,
                   application_launched=True if metrics["kernel_timing_available"] else (False if process is None else None),
                   application_executed=True if metrics["kernel_timing_available"] else (False if process is None else None), argv=command)
    # A started profiler alone cannot establish that it launched the application.
    (directory / "gpu-profile-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return combined_log, wall, metrics
