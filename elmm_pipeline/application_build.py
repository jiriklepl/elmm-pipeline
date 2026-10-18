"""Transport compiler-issued application build contracts through its public CLI.

The independent pipeline does not duplicate compiler command normalization or
infer source legality. The explicitly selected compiler verifies both phases.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .common import python_environment

PROTOCOL = "checked-interval-process-v4"
VERIFIER_TIMEOUT_SECONDS = 180


def selection_builds(manifest):
    records = [scope["exact_window"] for scope in manifest.get("scopes", []) if "exact_window" in scope]
    for record in records:
        if (not isinstance(record, dict) or record.get("contract", {}).get("protocol") != PROTOCOL
                or record["contract"].get("build_contract", {}).get("protocol") != "executed-application-build-v1"):
            raise ValueError("Exact-window cost evidence requires a supported executed application build contract")
    # A static native decision restores the original selection and emits no
    # selector. Its diagnostic contract must not force a CUDA build attestation.
    return [record for record in records if record.get("capture") or record.get("selection") == "resident"]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(transport, phase, request, directory):
    """Invoke only the selected snapshot's public verifier, without a shell."""
    root = Path(transport["compiler_root"]).resolve(strict=True)
    python = Path(transport["python"]).absolute()
    if not (root / "compiler/offload/application_build.py").is_file() or not python.is_file():
        raise ValueError("Application builds require the explicitly selected compiler verifier and Python")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_path, output = directory / (phase + "-request.json"), directory / (phase + ".json")
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    command = [str(python), "-m", "compiler.offload.application_build", phase,
               "--request", str(request_path), "--output", str(output)]
    environment = python_environment()
    environment["PYTHONPATH"] = str(root)
    record = {"argv": command, "cwd": str(root), "returncode": None,
              "timeout_seconds": VERIFIER_TIMEOUT_SECONDS, "timed_out": False}
    timeout_error = None
    try:
        result = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True,
                                check=False, timeout=VERIFIER_TIMEOUT_SECONDS)
        record["returncode"] = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired can retain byte output even with text=True. Preserve
        # that diagnostic stream before rejecting either verification phase.
        timeout_error = error
        record["timed_out"] = True
        stdout, stderr = (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                          for value in (error.stdout, error.stderr))
    (directory / (phase + "-command.json")).write_text(json.dumps(record, indent=2) + "\n")
    (directory / (phase + ".log")).write_text(stdout + stderr)
    if timeout_error is not None:
        raise ValueError("Compiler application build verification timed out after "
                         + str(VERIFIER_TIMEOUT_SECONDS) + " seconds; binary delivery blocked") from timeout_error
    if result.returncode:
        raise ValueError("Compiler application build verification failed; binary delivery blocked: " + result.stderr[-2000:])
    response = json.loads(result.stdout)
    value = json.loads(output.read_text())
    if response.get("status") != "verified" or response.get("identity") != value.get("identity"):
        raise ValueError("Compiler application verifier returned an inconsistent response")
    return value


def preflight(transport, build_dir):
    request = {key: transport[key] for key in ("manifest", "manifest_sha256", "artifact_root", "destinations")}
    request["build_root"] = str(Path(build_dir).resolve())
    return verify(transport, "preflight", request, Path(build_dir) / "application-build")


def attest(transport, token, receipt, build_dir):
    value = verify(transport, "attest", {"preflight": token, "receipt": str(receipt), "receipt_sha256": _hash(receipt)},
                   Path(build_dir) / "application-build")
    path = Path(build_dir).resolve() / "application-build" / "attest.json"
    return {"receipt": str(receipt), "receipt_sha256": _hash(receipt), "attestation": str(path),
            "attestation_sha256": _hash(path), "binary_sha256": value["binary_sha256"],
            "contract_identity": value["contract_identity"], "variant": value["variant"]}
