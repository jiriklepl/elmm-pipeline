"""Explicit source implementation switch for the qualified local DP backend.

This changes the selected solver implementation. It is neither the unchanged
original-wrapper contract nor a compiler effect/placement proof. Fresh source
analysis and a new numerical/observability qualification remain mandatory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .build_receipts import file_identity

_ORIGINAL_USE = "    use PoisFFT, PoisFFT_Solver => PoisFFT_Solver3D_DP"
_LAZY_USE = "    use elmm_gpu_pressure_lazy_poisfft, PoisFFT_Solver => PoisFFT_Solver3D_DP"
_CALLER = "poissonsolvers::poiss_poisfft"


def materialize_lazy_pressure_wrapper(source, directory, *, original_qualification):
    """Copy one authenticated original and change exactly its DP USE target.

    The result supports only a configured DPREC, non-PAR source export. The
    old CPU and original-v2 paths keep their source bytes and qualifications.
    This preparation never compiles, executes, selects a device or grants
    source definitions. No constructor/clock/debug statement is removed.
    """
    source = Path(source).resolve(strict=True)
    if (not isinstance(original_qualification, dict)
            or set(original_qualification) != {"path", "sha256"}
            or not isinstance(original_qualification["path"], str)
            or not Path(original_qualification["path"]).is_absolute()
            or file_identity(original_qualification["path"])["sha256"] != original_qualification["sha256"]):
        raise ValueError("Lazy pressure source requires its exact original qualification")
    qualified = json.loads(Path(original_qualification["path"]).read_text())
    caller = qualified["source_contract"]["caller"]
    original = file_identity(source)
    if caller["procedure"] != _CALLER or caller["sha256"] != original["sha256"]:
        raise ValueError("Lazy pressure source differs from the explicitly qualified original wrapper")
    text = source.read_text()
    first = text.index("  subroutine Poiss_PoisFFT(Phi,RHS)")
    last = text.index("  end subroutine Poiss_PoisFFT", first)
    body = text[first:last]
    if body.count(_ORIGINAL_USE) != 1:
        raise ValueError("Lazy pressure source requires one exact original DP USE target")
    changed = text[:first] + body.replace(_ORIGINAL_USE, _LAZY_USE, 1) + text[last:]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / source.name
    with destination.open("x") as stream:
        stream.write(changed)
    implementation = Path(__file__).with_name("gpu_pressure_sources") / "gpu_pressure_lazy_poisfft.f90"
    record = {
        "schema_version": 1, "kind": "explicit_lazy_cpu_poisfft_source_implementation",
        "entry": _CALLER, "original": original, "replacement": file_identity(destination),
        "original_qualification": dict(original_qualification), "implementation": file_identity(implementation),
        "edit": {"old": _ORIGINAL_USE, "new": _LAZY_USE, "occurrences": 1},
        "configured_source_requirement": {"required_defines": ["DPREC"], "forbidden_defines": ["PAR"]},
        "constructor": "same original call syntax; different explicitly selected metadata-only receiver implementation",
        "cpu_fallback": "first reached native Execute creates pinned CPU plan; its cost moves inside that solve timer",
        "gpu_plan": "existing checked outer owner/provider slot; no global GPU cache",
        "source_authority": False, "placement_authority": False, "performance_claim": False,
        "required_qualification": "fresh configured source and explicit new numerical/receiver-observability envelope"}
    (directory / "source-implementation.json").write_text(json.dumps(record, indent=2) + "\n")
    if file_identity(source) != original:
        raise ValueError("Original pressure source changed during preparation")
    if hashlib.sha256(Path(original_qualification["path"]).read_bytes()).hexdigest() != original_qualification["sha256"]:
        raise ValueError("Original pressure qualification changed during preparation")
    return record


def prepare_lazy_poisfft_contract(source_paths, analysis_sources, directory, *, backend, fortran,
                                  numerical_qualification, opaque_qualification, workspace_cap_bytes,
                                  required_entry_read_prerequisite=None, provider_only_lifetime=None,
                                  provider_retention_prerequisite=None):
    """Issue fresh inputs for the separately qualified explicit lazy source.

    No original-v2 qualification is implicitly reused for changed source.
    The generic compiler still issues the exact typed receiver/CALL proof.
    An optional exact reference to a reviewed pinned-library must-read premise
    binds RHS interior requirements and all six descriptor/query equalities to
    fresh source inputs. It grants no incoming payload, halo or lifetime facts;
    neither numerical agreement nor INTENT(IN) supplies this premise.
    An optional provider_only_lifetime call-path request selects an explicit
    native parent. The compiler must authenticate every original hop and emit
    complete per-call field publication before it can retain private plans.
    """
    from .gpu_pressure_build import require_pressure_backend
    from .gpu_pressure_contract import _prepare_poisfft_contract, _reference

    if not isinstance(analysis_sources, dict):
        raise TypeError("Pressure source preparation requires an authentic configured-source export")
    opaque = _reference(opaque_qualification)
    selected = require_pressure_backend(backend, fortran=fortran)
    lazy = selected["requirements"].get("cpu_fallback")
    canonical = Path(__file__).with_name("gpu_pressure_sources") / "gpu_pressure_lazy_poisfft.f90"
    source_inputs = analysis_sources.get("source_inputs")
    # The receiver's source coordinates belong to its issued qualification and
    # configured export, not to this tool installation. Preserve that identity;
    # equal bytes never authorize rewriting arbitrary source-coordinate maps.
    qualified = opaque.get("implementation") if isinstance(opaque, dict) else None
    packaged = lazy.get("source") if isinstance(lazy, dict) else None
    def exact_source(reference):
        if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()):
            return False
        try:
            return file_identity(reference["path"]) == reference
        except (OSError, ValueError):
            return False
    if (not isinstance(opaque, dict) or opaque.get("kind") != "explicit_lazy_cpu_poisfft_receiver_qualification"
            or not exact_source(qualified) or not exact_source(packaged)
            or qualified["sha256"] != file_identity(canonical)["sha256"]
            or packaged["sha256"] != qualified["sha256"]
            or lazy.get("implementation") != "lazy_poisfft_v1"
            or packaged["path"] not in selected["extra_sources"]
            or not isinstance(source_inputs, dict)
            or source_inputs.get(qualified["path"]) != qualified["sha256"]):
        raise ValueError("Lazy pressure contract requires its exact separately qualified source implementation")
    return _prepare_poisfft_contract(source_paths, analysis_sources, directory, backend=backend, fortran=fortran,
        numerical_qualification=numerical_qualification, opaque_qualification=opaque_qualification,
        workspace_cap_bytes=workspace_cap_bytes, callee="elmm_gpu_pressure_lazy_poisfft::lazy_execute",
        required_entry_read_prerequisite=required_entry_read_prerequisite,
        provider_only_lifetime=provider_only_lifetime,
        provider_retention_prerequisite=provider_retention_prerequisite)
