"""Validate application completion and numeric outputs without NumPy."""
from array import array
import json
import math
from pathlib import Path
import re
import sys


def validate_log(text, expected_max_steps=None, *, expected_completed_steps=None):
    failures = ["ELMM Abort:", "Solution diverged.", "Maximum clock time exceeded.",
                "Fortran runtime error", "ERROR STOP", "SIGSEGV", "Floating point exception"]
    for marker in failures:
        if marker.lower() in text.lower():
            raise ValueError(f"Application reported failure: {marker}")
    steps = [int(s) for s in re.findall(r"tstep:\s*(\d+)", text)]
    if not steps or max(steps) <= 0 or not re.search(r"(?m)^\s*saved\.\s*$", text):
        raise ValueError("Application did not complete timesteps and save final results")
    natural_end = "Time limit reached." in text or "Steady state reached." in text
    if not natural_end and (expected_max_steps is None or max(steps) != expected_max_steps):
        raise ValueError("Application stopped before the requested step/time limit")
    if expected_completed_steps is not None and max(steps) != expected_completed_steps:
        raise ValueError(f"Workload differs from baseline: completed {max(steps)} timesteps, "
                         f"expected {expected_completed_steps}")
    times = {}
    for name, label in (("time_steps_seconds", "Total wall clock time for time steps"),
                        ("poisson_seconds", "Wall clock time for Poisson solver")):
        match = re.search(re.escape(label) + r":\s*([\d.]+)", text)
        times[name] = float(match[1]) if match else None
    if times["time_steps_seconds"] is None:
        raise ValueError("Missing application timestep timing")
    return {"completed_steps": max(steps), "natural_end": natural_end, **times}


def read_vtk(path, *, require_flow_fields=True):
    fields = {}
    with Path(path).open("rb") as stream:
        def line():
            return stream.readline().decode("ascii").strip()

        def values(count, dtype):
            if dtype not in {"float", "double"}:
                raise ValueError(f"Unsupported VTK type: {dtype}")
            data = array("f" if dtype == "float" else "d")
            raw = stream.read(count * data.itemsize)
            if len(raw) != count * data.itemsize:
                raise ValueError("Truncated VTK data")
            data.frombytes(raw)
            if sys.byteorder == "little":
                data.byteswap()
            if not all(math.isfinite(v) for v in data):
                raise ValueError("Non-finite VTK field")
            if stream.read(1) != b"\n":
                raise ValueError("Missing VTK field delimiter")
            return data

        if not line().startswith("# vtk DataFile"):
            raise ValueError("Not a legacy VTK file")
        line()
        if line() != "BINARY" or line() != "DATASET RECTILINEAR_GRID":
            raise ValueError("Expected binary RECTILINEAR_GRID VTK")
        dimensions = line().split()
        if dimensions[0] != "DIMENSIONS":
            raise ValueError("Missing VTK dimensions")
        shape = tuple(map(int, dimensions[1:]))
        for label, n in zip(("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"), shape):
            parts = line().split()
            if parts[0] != label or int(parts[1]) != n:
                raise ValueError("Unexpected VTK coordinates")
            fields[label] = values(n, parts[2])
        parts = line().split()
        count = math.prod(shape)
        if parts != ["POINT_DATA", str(count)]:
            raise ValueError("Unexpected VTK point count")
        while True:
            header = line()
            if not header:
                if stream.peek(1) if hasattr(stream, "peek") else False:
                    continue
                break
            parts = header.split()
            if parts[0] == "SCALARS":
                components = int(parts[3]) if len(parts) == 4 else 1
                if not line().startswith("LOOKUP_TABLE"):
                    raise ValueError("Missing VTK scalar lookup table")
            elif parts[0] == "VECTORS":
                components = 3
            else:
                raise ValueError(f"Unsupported VTK declaration: {header}")
            if parts[1] in fields:
                raise ValueError("Duplicate VTK field")
            fields[parts[1]] = values(count * components, parts[2])
    if require_flow_fields and ("u" not in fields or "p" not in fields):
        raise ValueError("Validation requires final velocity u and pressure p fields")
    if len(fields) <= 3:
        raise ValueError("VTK requires at least one numeric field")
    return shape, fields


def compare_vtk(reference, candidate, *, atol=1e-6, rtol=1e-5):
    if any(not math.isfinite(t) or t < 0 for t in (atol, rtol)):
        raise ValueError("Validation tolerances must be finite and nonnegative")
    ref_shape, ref_fields = reference
    shape, fields = candidate
    if shape != ref_shape or fields.keys() != ref_fields.keys():
        raise ValueError("Final-state field names or dimensions differ")
    checks = {}
    for name, expected in ref_fields.items():
        actual = fields[name]
        if len(actual) != len(expected):
            raise ValueError(f"Field length mismatch: {name}")
        worst, failures = 0.0, 0
        for a, b in zip(expected, actual):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Non-finite comparison field: {name}")
            error = abs(a - b)
            worst = max(worst, error)
            if error > atol + rtol * abs(a):
                failures += 1
        checks[name] = {"max_absolute_error": worst, "failed_values": failures,
                        "values": len(expected)}
    return {"passed": all(c["failed_values"] == 0 for c in checks.values()),
            "atol": atol, "rtol": rtol, "fields": checks,
            "note": "ELMM final VTK values are float32, including double-precision builds."}


def validate_generic_log(text, *, mode="json-stdout", expected_completed_steps=None):
    """Validate completion of a generic Fortran application run."""
    failures = [
        "ELMM Abort:",
        "Fortran runtime error",
        "ERROR STOP",
        "SIGSEGV",
        "Floating point exception",
    ]
    for marker in failures:
        if marker.lower() in text.lower():
            raise ValueError(f"Application reported failure: {marker}")

    if mode == "json-stdout":
        prefix = "FORT_APPLICATION_COMPLETION "
        matches = [
            line[len(prefix) :].strip()
            for line in text.splitlines()
            if line.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError("Application output must contain exactly one FORT_APPLICATION_COMPLETION JSON line")
        try:
            payload = json.loads(matches[0])
        except ValueError as error:
            raise ValueError("Malformed FORT_APPLICATION_COMPLETION JSON line") from error
        if not isinstance(payload, dict):
            raise ValueError("FORT_APPLICATION_COMPLETION payload must be a JSON object")
        steps = payload.get("completed_steps", expected_completed_steps if expected_completed_steps is not None else 1)
        if expected_completed_steps is not None and steps != expected_completed_steps:
            raise ValueError(
                f"Workload differs from baseline: completed {steps} steps, expected {expected_completed_steps}"
            )
        return {
            "completed_steps": steps,
            "natural_end": payload.get("natural_end", True),
            "time_steps_seconds": payload.get("time_steps_seconds"),
            "poisson_seconds": payload.get("poisson_seconds"),
            **{k: v for k, v in payload.items() if k not in {"completed_steps", "natural_end", "time_steps_seconds", "poisson_seconds"}},
        }

    return {
        "completed_steps": expected_completed_steps if expected_completed_steps is not None else 1,
        "natural_end": True,
        "time_steps_seconds": None,
        "poisson_seconds": None,
    }


def _read_numeric_values(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".bin":
        raw = path.read_bytes()
        if len(raw) % 8 != 0:
            raise ValueError(f"Binary float64 file length is not a multiple of 8: {path}")
        data = array("d")
        data.frombytes(raw)
        if not all(math.isfinite(v) for v in data):
            raise ValueError(f"Non-finite value in binary file: {path}")
        return list(data)
    text = path.read_text(encoding="utf-8")
    tokens = [tok for tok in re.split(r"[\s,;]+", text.strip()) if tok]
    values = [float(tok) for tok in tokens]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Missing or non-finite numeric values in {path}")
    return values


def compare_numeric_files(reference_dir, candidate_dir, files, *, atol=1e-6, rtol=1e-5):
    """Compare declared numeric output files (.vtk, .bin, or text) within tolerance."""
    if any(not math.isfinite(t) or t < 0 for t in (atol, rtol)):
        raise ValueError("Validation tolerances must be finite and nonnegative")
    ref_root = Path(reference_dir).resolve(strict=True)
    cand_root = Path(candidate_dir).resolve(strict=True)
    if not files:
        raise ValueError("Generic file comparison requires at least one declared output file")

    checks = {}
    for rel in files:
        ref_path = (ref_root / rel).resolve()
        cand_path = (cand_root / rel).resolve()
        if not ref_path.is_relative_to(ref_root) or not cand_path.is_relative_to(cand_root):
            raise ValueError(f"Output file escapes run directory: {rel}")
        if not ref_path.is_file() or not cand_path.is_file():
            raise ValueError(f"Missing output file for validation: {rel}")
        if ref_path.suffix.lower() == ".vtk":
            result = compare_vtk(
                read_vtk(ref_path, require_flow_fields=False),
                read_vtk(cand_path, require_flow_fields=False),
                atol=atol,
                rtol=rtol,
            )
            checks[str(rel)] = result
        else:
            expected = _read_numeric_values(ref_path)
            actual = _read_numeric_values(cand_path)
            if len(actual) != len(expected):
                raise ValueError(f"Output length mismatch for {rel}: {len(actual)} vs {len(expected)}")
            worst, failures = 0.0, 0
            for a, b in zip(expected, actual):
                error = abs(a - b)
                worst = max(worst, error)
                if error > atol + rtol * abs(a):
                    failures += 1
            checks[str(rel)] = {
                "passed": failures == 0,
                "max_absolute_error": worst,
                "failed_values": failures,
                "values": len(expected),
            }
    return {
        "passed": all(entry["passed"] for entry in checks.values()),
        "atol": atol,
        "rtol": rtol,
        "fields": checks,
    }
