"""Link explicit application validation drivers; compare every dumped element."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess

import numpy as np

from .identity import read, write
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
    if actual.keys() != reference.keys():
        raise ValueError('missing complete fields')
    comparisons = {}
    for name, (values, layout) in actual.items():
        expected, original_layout = reference[name]
        if values.shape != expected.shape or layout != original_layout:
            raise ValueError('complete field shape or original bounds mismatch')
        comparisons[name] = {'values': int(values.size), 'passed': bool(np.allclose(
            values, expected, rtol=driver['rtol'], atol=driver['atol'], equal_nan=False)),
            'max_absolute_error': float(np.max(np.abs(values - expected), initial=0))}
        if layout is not None:
            comparisons[name]['layout'] = layout
    return comparisons


def run(root):
    root = Path(root).resolve(strict=True)
    state = check(root)
    output = root / "halos"
    output.mkdir(exist_ok=False)
    if not state["halo_drivers"]:
        raise ValueError("explicit complete-array validation drivers required")
    report = {"status": "in_progress", "drivers": [], "performance_measured": False}
    for driver in state["halo_drivers"]:
        reference = None
        entry = {"name": driver["name"], "coverage": driver["coverage"], "runs": []}
        report["drivers"].append(entry)
        modes = sorted(driver["sources"], key=lambda name: name != "native")
        if "native" not in modes:
            raise ValueError("each validation driver requires an original native reference")
        for mode in modes:
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
                env = environment(state["threads"], traced=mode != "native",
                                  cpu_placement=state.get("cpu_placement"))
                env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
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
            entry["runs"].append({"mode": mode, "comparisons": comparisons})
            write(output / "report.json", report)
            if not all(row["passed"] for row in comparisons.values()):
                raise RuntimeError("complete-array or halo disagreement")
    check(root)
    report["status"] = "passed"
    write(output / "report.json", report)
