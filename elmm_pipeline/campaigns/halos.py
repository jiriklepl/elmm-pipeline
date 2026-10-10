"""Link explicit application validation drivers; compare every dumped element."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from .identity import write
from .launch_environment import launch_environment_values
from .prepare import check
from .run import environment, run_one


def fields(directory, driver):
    """Read explicit native-precision dumps, optionally with original bounds."""
    result = {}
    dtype = np.dtype(driver['dtype'])
    metadata = driver.get('metadata_format')
    if metadata not in (None, 'storage-shape-bounds-v1'):
        raise ValueError('unsupported complete-field metadata format')
    for path in directory.glob('fields-*.bin'):
        values = np.fromfile(path, dtype=dtype)
        if path.stat().st_size != values.size * dtype.itemsize:
            raise ValueError('partial element in complete field')
        layout = None
        if metadata:
            parts = [int(value) for value in path.with_suffix('.shape').read_text().split()]
            rank, remainder = divmod(len(parts)-1, 3)
            if remainder or not 1 <= rank <= 15 or parts[0] != dtype.itemsize * 8:
                raise ValueError('complete field precision or metadata rank mismatch')
            shape, lower, upper = (parts[1+index*rank:1+(index+1)*rank] for index in range(3))
            if (any(extent < 0 or extent != max(0, hi-lo+1)
                    for extent, lo, hi in zip(shape, lower, upper, strict=True))
                    or math.prod(shape) != values.size):
                raise ValueError('complete field extent or byte count mismatch')
            layout = {'bits': parts[0], 'shape': shape, 'lower_bounds': lower, 'upper_bounds': upper}
        result[path.name] = (values, layout)
    if len(result) != driver['field_count']:
        raise ValueError('missing complete fields')
    return result


def compare_fields(actual, reference, driver):
    if any(not math.isfinite(driver[name]) or driver[name] < 0 for name in ('atol', 'rtol')):
        raise ValueError('validation tolerances must be finite and nonnegative')
    if actual.keys() != reference.keys():
        raise ValueError('missing complete fields')
    comparisons = {}
    for name, (values, layout) in actual.items():
        expected, original_layout = reference[name]
        if values.shape != expected.shape or layout != original_layout:
            raise ValueError('complete field shape or original bounds mismatch')
        finite_actual, finite_reference = np.isfinite(values), np.isfinite(expected)
        # Nonfinite values are always failures, including matching infinities.
        # Keep the failure receipt serializable even when subtraction overflows.
        with np.errstate(over='ignore', invalid='ignore'):
            matches = (finite_actual & finite_reference & np.isclose(
                values, expected, rtol=driver['rtol'], atol=driver['atol'], equal_nan=False))
            errors = np.abs(values - expected)
        failed = np.flatnonzero(~matches)
        finite_errors = np.isfinite(errors)
        comparisons[name] = {'values': int(values.size), 'passed': not bool(failed.size),
            'failed_values': int(failed.size),
            'candidate_nonfinite_values': int(np.count_nonzero(~finite_actual)),
            'reference_nonfinite_values': int(np.count_nonzero(~finite_reference)),
            'absolute_error_nonfinite_values': int(np.count_nonzero(~finite_errors)),
            'max_absolute_error': float(np.max(errors, initial=0)) if np.all(finite_errors) else None,
            'failed_flat_indices': failed[:8].tolist(),
            'failed_locations_truncated': bool(failed.size > 8),
            'atol': driver['atol'], 'rtol': driver['rtol']}
        if layout is not None:
            comparisons[name]['layout'] = layout
    return comparisons


def run(root):
    root = Path(root).resolve(strict=True)
    state = check(root)
    output = root / "halos"
    output.mkdir(exist_ok=False)
    report = {"status": "in_progress", "drivers": [], "failures": [], "performance_measured": False}
    write(output / "report.json", report)
    try:
        if not state["halo_drivers"]:
            raise ValueError("explicit complete-array validation drivers required")
        for driver in state["halo_drivers"]:
            _run_driver(root, state, output, driver, report)
        check(root)
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write(output / "report.json", report)
        raise
    report["status"] = "failed" if report["failures"] else "passed"
    write(output / "report.json", report)
    if report["failures"]:
        raise RuntimeError("complete-array or halo disagreement; all comparisons retained")


def _run_driver(root, state, output, driver, report):
    reference = None
    entry = {"name": driver["name"], "coverage": driver["coverage"], "runs": []}
    report["drivers"].append(entry)
    modes = sorted(driver["sources"], key=lambda name: name != "native")
    if "native" not in modes:
        raise ValueError("each validation driver requires an original native reference")
    for mode in modes:
        env = environment(state["threads"], traced=mode != "native",
                          cpu_placement=state.get("cpu_placement"),
                          openmp_launch_environment=state.get("openmp_launch_environment"))
        env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
        directory = output / driver["name"] / mode
        directory.mkdir(parents=True)
        binary = directory / "driver"
        link = driver["link"][mode]
        command = [state["tools"]["fortran"]["path"], *driver["flags"],
                   *["-I" + path for path in link["include_dirs"]],
                   str(root / driver["sources"][mode]), *link["objects"], *link["flags"], "-o", str(binary)]
        write(directory / "build-command.json", {"argv": command})
        compiled = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=False, timeout=120)
        (directory / "build.log").write_text(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            raise RuntimeError("validation driver build failed: " + str(directory))
        if 'application_case' in driver:
            case = next(row for row in state['cases'] if row['name'] == driver['application_case'])
            original_mode = next(row for row in state['modes'] if row['name'] == mode)
            dump_directory = directory / 'run'
            run_one(root, state, case, {**original_mode, 'binary': str(binary)}, dump_directory, traced=True)
        else:
            result = subprocess.run([str(binary)], cwd=directory, env=env,
                text=True, capture_output=True, check=False, timeout=120)
            (directory / "stdout.log").write_text(result.stdout)
            (directory / "stderr.log").write_text(result.stderr)
            if result.returncode or driver["success_marker"] not in result.stdout:
                raise RuntimeError("validation driver failed: " + str(directory))
            dump_directory = directory
        observed = fields(dump_directory, driver)
        if reference is None:
            reference = observed
        comparisons = compare_fields(observed, reference, driver)
        entry["runs"].append({"mode": mode, "comparisons": comparisons,
                              "openmp_launch_environment": launch_environment_values(env)})
        failed = [name for name, row in comparisons.items() if not row["passed"]]
        if failed:
            report["failures"].append({"driver": driver["name"], "mode": mode, "fields": failed})
        write(output / "report.json", report)
