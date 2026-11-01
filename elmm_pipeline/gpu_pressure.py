"""Explicit stream-aware pressure adapter artifacts, without source authority.

The compiler must independently authenticate the original call, numerical
configuration, canonical physical roots, lifetimes and reached effects. Merely
constructing this metadata never admits an ELMM procedure or chooses a device.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

_SOURCE = Path(__file__).with_name("gpu_pressure_sources")
_FILES = ("gpu_pressure.h", "gpu_pressure.cpp", "gpu_pressure_halo.cu", "gpu_pressure.f90")
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1
_SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


def _integers(values, count, lower, upper, label):
    result = tuple(values)
    if len(result) != count or any(type(value) is not int or not lower <= value <= upper for value in result):
        raise ValueError(f"Invalid {label}")
    return result


@dataclass(frozen=True)
class PressureProblem:
    """Validated numeric/layout requirements; never a compiler proof token."""

    sizes: tuple[int, int, int]
    lengths: tuple[float, float, float]
    boundary_conditions: tuple[int, int, int, int, int, int]
    approximation: int
    workspace_cap_bytes: int
    phi_lower_bounds: tuple[int, int, int] = (-1, -1, -1)
    rhs_lower_bounds: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        sizes = _integers(self.sizes, 3, 2, (1 << 31) - 5, "pressure extents")
        lengths = tuple(self.lengths)
        try:
            if len(lengths) != 3 or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                                       for value in lengths):
                raise ValueError("Invalid pressure lengths")
        except OverflowError as error:
            raise ValueError("Invalid pressure lengths") from error
        bcs = _integers(self.boundary_conditions, 6, 0, 4, "pressure boundary conditions")
        if any(bcs[2 * axis] not in (0, 4) or bcs[2 * axis] != bcs[2 * axis + 1] for axis in range(3)):
            raise ValueError("Only periodic/NeumannStag boundary pairs are qualified")
        if type(self.approximation) is not int or self.approximation not in (2, 4):
            raise ValueError("Only FD2 and FD4 are qualified")
        _integers((self.workspace_cap_bytes,), 1, 1, _SIZE_MAX, "workspace cap")
        phi_lower = _integers(self.phi_lower_bounds, 3, _I64_MIN, _I64_MAX, "Phi lower bounds")
        rhs_lower = _integers(self.rhs_lower_bounds, 3, _I64_MIN, _I64_MAX, "RHS lower bounds")
        for shape, lower in ((tuple(n + 4 for n in sizes), phi_lower), (tuple(n + 2 for n in sizes), rhs_lower)):
            if math.prod(shape) > _SIZE_MAX // 8 or any(lo > _I64_MAX - extent + 1 for lo, extent in zip(lower, shape)):
                raise ValueError("Pressure descriptor extent/coordinate overflow")
        for name, value in (("sizes", sizes), ("lengths", tuple(float(x) for x in lengths)),
                            ("boundary_conditions", bcs), ("phi_lower_bounds", phi_lower),
                            ("rhs_lower_bounds", rhs_lower)):
            object.__setattr__(self, name, value)

    def effects(self):
        lower, upper = (2, 2, 2), tuple(n + 2 for n in self.sizes)
        faces, reads = [], []
        for axis in range(3):
            for side in range(2):
                lo, hi = list(lower), list(upper)
                lo[axis] = upper[axis] if side else 0
                hi[axis] = lo[axis] + 2
                faces.append({"lower": lo, "upper": hi})
                lo, hi = list(lower), list(upper)
                periodic = self.boundary_conditions[2 * axis + side] == 0
                lo[axis] = upper[axis] - 2 if periodic != bool(side) else 2
                hi[axis] = lo[axis] + 2
                reads.append({"lower": lo, "upper": hi})
        interior = {"lower": list(lower), "upper": list(upper)}
        return {
            "solve": {"phi": {"reads": [], "writes": [interior], "overwrites": [interior]},
                      "rhs": {"reads": [{"lower": [1, 1, 1], "upper": [n + 1 for n in self.sizes]}],
                              "writes": [], "overwrites": []}},
            "halo": {"phi": {"reads": reads, "writes": faces, "overwrites": faces}},
        }

    def record(self):
        return {
            "schema_version": 1, "kind": "explicit_gpu_pressure_requirements",
            "source_authority": False, "placement_authority": False, "estimates_available": False,
            "provider_abi": "poisson_solver_device_v1", "runtime_abi": "fort_scope_external_v1",
            "equation": "laplacian(phi)=rhs", "dimensions": 3, "element_type": "real64",
            "sizes": list(self.sizes), "lengths": list(self.lengths),
            "boundary_conditions": list(self.boundary_conditions), "approximation": self.approximation,
            "phi": {"extents": [n + 4 for n in self.sizes], "lower_bounds": list(self.phi_lower_bounds),
                    "ghosts": [2, 2, 2], "dense_first_index_fastest": True},
            "rhs": {"extents": [n + 2 for n in self.sizes], "lower_bounds": list(self.rhs_lower_bounds),
                    "ghosts": [1, 1, 1], "dense_first_index_fastest": True},
            "workspace_cap_bytes": self.workspace_cap_bytes, "effects": self.effects(),
            "lifetime": "explicit caller-owned domain; stable canonical storage until its checked retirement",
            "errors": "started failure forbids replay; provider destruction precedes workspace release",
            "compiler_requirements": ["original call/body identity and numerical configuration",
                                      "canonical physical root/generation and complete dense layout",
                                      "source-proven reached input definitions and exact output effects",
                                      "storage/alias/optional/allocation guards and cleanup frontiers"],
        }


def source_manifest():
    return [{"path": str(_SOURCE / name), "sha256": hashlib.sha256((_SOURCE / name).read_bytes()).hexdigest()}
            for name in _FILES]


def prepare_adapter(problem: PressureProblem, directory: str | Path):
    """Copy the exact adapter sources to a fresh directory; run no compiler."""
    if not isinstance(problem, PressureProblem):
        raise TypeError("PressureProblem required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sources = []
    for entry in source_manifest():
        destination = directory / Path(entry["path"]).name
        shutil.copyfile(entry["path"], destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Pressure adapter source changed during preparation")
        sources.append({"path": str(destination.resolve()), "sha256": entry["sha256"]})
    result = {**problem.record(), "sources": sources,
              "build_requirements": {"cxx_standard": 17, "cuda_stream": "supplied by external callback",
                                     "include_roots": ["scoped_runtime.h", "poisson-solver/device_api.h"],
                                     "link": ["one shared scoped_runtime object", "poisson-solver status API library"]}}
    (directory / "pressure-adapter.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
