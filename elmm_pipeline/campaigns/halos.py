"""Link explicit application validation drivers; compare every dumped element."""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from .identity import read, write
from .prepare import check
from .run import environment


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
            env = environment(state["threads"], traced=mode != "native")
            env["CUDA_VISIBLE_DEVICES"] = state["selected_gpu_uuid"]
            result = subprocess.run([str(binary)], cwd=directory, env=env,
                text=True, capture_output=True, check=False, timeout=120)
            (directory / "stdout.log").write_text(result.stdout)
            (directory / "stderr.log").write_text(result.stderr)
            if result.returncode or driver["success_marker"] not in result.stdout:
                raise RuntimeError("validation driver failed: " + str(directory))
            fields = {path.name: np.fromfile(path, dtype=driver["dtype"])
                      for path in directory.glob("fields-*.bin")}
            if reference is None:
                reference = fields
            if len(fields) != driver["field_count"] or set(fields) != set(reference):
                raise ValueError("missing complete fields")
            comparisons = {}
            for name, values in fields.items():
                expected = reference[name]
                if values.shape != expected.shape:
                    raise ValueError("complete field shape mismatch")
                comparisons[name] = {"values": int(values.size), "passed": bool(np.allclose(
                    values, expected, rtol=driver["rtol"], atol=driver["atol"], equal_nan=False)),
                    "max_absolute_error": float(np.max(np.abs(values - expected), initial=0))}
            entry["runs"].append({"mode": mode, "comparisons": comparisons})
            write(output / "report.json", report)
            if not all(row["passed"] for row in comparisons.values()):
                raise RuntimeError("complete-array or halo disagreement")
    check(root)
    report["status"] = "passed"
    write(output / "report.json", report)
