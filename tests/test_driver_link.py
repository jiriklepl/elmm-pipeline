"""Source-selected diagnostics execute real commands and preserve their recipes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from elmm_pipeline import build_receipts
from elmm_pipeline.build_receipts import CommandRecorder, file_identity, summarize_build
from elmm_pipeline.campaigns.driver_link import build_driver_link


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return file_identity(path)


@pytest.fixture
def application(tmp_path):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("GNU Fortran required for executed relink tests")
    cwd, build, immutable = (tmp_path / name for name in ("application source", "build output", "immutable"))
    for path in (cwd, build, immutable, build / "P", build / "module path"):
        path.mkdir()
    original, destination = immutable / "driver.f90", cwd / "selected driver.f90"
    original.write_text("program driver\nuse values\ninteger::x\nx=2\nprint '(I0)',x\nend\n")
    destination.write_bytes(original.read_bytes())
    helper = cwd / "values.f90"
    helper.write_text("module values\ninteger,parameter::unused=1\nend module\n")
    helper_obj, binary = build / "P/main.o", build / "application"
    flags = ["-O1", "-O0", "-fopenmp", "-I", "../build output/module path", "-J../build output/module path"]
    commands = build / "executed"
    recorder = CommandRecorder(commands)
    compile_argv = [compiler, *flags, "-o", "../build output/entry-point.o", "-c", destination.name]
    link_argv = [compiler, "-fopenmp", "../build output/P/main.o", "../build output/entry-point.o",
                 "-Wl,--disable-new-dtags", "-lm", "-o", "../build output/application"]
    for argv in ([compiler, *flags, "-c", helper.name, "-o", str(helper_obj)], compile_argv, link_argv):
        assert recorder.run(argv, cwd=cwd, timeout=240).returncode == 0
    receipt = summarize_build(binary, [commands], {str(p): file_identity(p)["sha256"] for p in (destination, helper)})
    receipt_ref = save(build / "receipts.json", receipt)
    # This transport fixture is deliberately not a compiler attestation. The
    # separate public-verifier test below uses actual compiler-issued contracts.
    attestation = {"receipt": receipt_ref["path"], "receipt_sha256": receipt_ref["sha256"],
                   "receipt_identity": receipt["identity"], "binary": str(binary),
                   "binary_sha256": file_identity(binary)["sha256"], "contract_identity": "transport-fixture",
                   "variant": "resident", "preflight": {"request": {
                       "destinations": {str(original): str(destination)}}}}
    attestation_ref = save(build / "attestation.json", attestation)
    observation = {"receipt": receipt_ref["path"], "receipt_sha256": receipt_ref["sha256"],
                   "attestation": attestation_ref["path"], "attestation_sha256": attestation_ref["sha256"],
                   **{key: attestation[key] for key in ("binary_sha256", "contract_identity", "variant")}}
    source = tmp_path / "diagnostic driver.f90"
    source.write_text(original.read_text().replace("print '(I0)',x", "print '(A,I0)','diagnostic=',x"))
    arguments = {"original_source": original, "source": source, "object_path": tmp_path / "diagnostic.o",
                 "output": tmp_path / "diagnostic program", "directory": tmp_path / "diagnostic evidence",
                 "module_inputs": [build / "module path/values.mod"]}
    return observation, arguments, receipt


def test_real_relink_preserves_order_relative_paths_unrelated_objects_and_modules(application):
    observation, arguments, receipt = application
    diagnostic = build_driver_link(observation, **arguments)
    assert set(diagnostic) == {"plan", "build", "commands"}
    plan = json.loads(Path(diagnostic["plan"]["path"]).read_text())
    commands = [json.loads(Path(item["path"]).read_text()) for item in diagnostic["commands"]]
    unit = next(unit for unit in receipt["units"] if Path(unit["object"]).name == "entry-point.o")
    expected_compile = list(unit["execution"]["argv"])
    expected_compile[-1] = str(arguments["source"])
    expected_compile[expected_compile.index("-o") + 1] = str(arguments["object_path"])
    expected_link = list(receipt["link"]["argv"])
    expected_link[expected_link.index("../build output/entry-point.o")] = str(arguments["object_path"])
    expected_link[-1] = str(arguments["output"])
    assert commands[0]["argv"] == plan["relink"]["compile_argv"] == expected_compile
    assert commands[1]["argv"] == plan["relink"]["link_argv"] == expected_link
    assert "../build output/P/main.o" in expected_link
    assert all(row["timeout_seconds"] == 240 and row["timed_out"] is False for row in commands)
    assert all(row["status"] == "completed" and row["returncode"] == 0 for row in commands)
    module = str(arguments["module_inputs"][0])
    assert all(module in {item["path"] for item in row["inputs"]} for row in commands)
    assert subprocess.run([str(arguments["output"])], text=True, capture_output=True,
                          check=True, timeout=5).stdout.strip() == "diagnostic=2"
    assert subprocess.run([receipt["binary"]], text=True, capture_output=True, check=True, timeout=5).stdout.strip() == "2"
    assert build_receipts.require_build_receipts(observation["receipt"]) == receipt
    assert json.loads((arguments["directory"] / "observation.json").read_text()) == diagnostic
    with pytest.raises(ValueError, match="already exists"):
        build_driver_link(observation, **arguments)


@pytest.mark.parametrize("change,reason", [("source", "unchanged original"), ("object", "file changed"),
                                           ("receipt", "observation changed"), ("environment", "environment differs"),
                                           ("module", "module inputs"), ("selection", "unchanged original")])
def test_changed_or_unmapped_inputs_reject_before_execution(application, change, reason):
    observation, arguments, receipt = application
    if change == "source":
        arguments["original_source"].write_text("changed original source\n")
    elif change == "object":
        Path(receipt["units"][0]["object"]).write_text("changed object\n")
    elif change == "receipt":
        Path(observation["receipt"]).write_text("{}")
    elif change == "environment":
        arguments["environment"] = {**os.environ, "FCFLAGS": "-ffast-math"}
    elif change == "module":
        arguments["module_inputs"] = [arguments["source"]]
    else:
        arguments["original_source"] = arguments["source"]
    with pytest.raises(ValueError, match=reason):
        build_driver_link(observation, **arguments)
    assert not arguments["directory"].exists()
    assert not arguments["object_path"].exists()
    assert not arguments["output"].exists()


@pytest.mark.parametrize("phase", ["compile", "link"])
def test_failed_real_build_preserves_evidence_and_never_returns_observation(application, phase):
    observation, arguments, _receipt = application
    arguments["source"].write_text("invalid Fortran syntax\n" if phase == "compile" else
                                   "program driver\ncall missing_external_procedure()\nend\n")
    with pytest.raises(ValueError, match="diagnostic " + phase + " failed"):
        build_driver_link(observation, **arguments)
    directory = arguments["directory"]
    assert (directory / (phase + ".log")).read_text()
    assert json.loads((directory / "build.json").read_text())["status"] == "failed"
    rows = [json.loads(path.read_text()) for path in (directory / "commands").glob("*.json")]
    assert len(rows) == (1 if phase == "compile" else 2)
    assert any(row["returncode"] != 0 for row in rows)
    assert not (directory / "observation.json").exists()


def test_timeout_retains_command_and_partial_log_and_blocks_relink(application, monkeypatch):
    observation, arguments, _receipt = application
    run = subprocess.run

    def timeout(argv, **kwargs):
        if "--version" in argv:
            return run(argv, **kwargs)
        assert kwargs["timeout"] == 240
        kwargs["stdout"].write(b"compiler partial diagnostic\n")
        raise subprocess.TimeoutExpired(argv, 240)

    monkeypatch.setattr(build_receipts.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        build_driver_link(observation, **arguments)
    directory = arguments["directory"]
    row, = [json.loads(path.read_text()) for path in (directory / "commands").glob("*.json")]
    assert row["status"] == "failed" and row["timed_out"] is True and row["timeout_seconds"] == 240
    assert (directory / "compile.log").read_text() == "compiler partial diagnostic\n"
    assert json.loads((directory / "build.json").read_text())["status"] == "failed"
    assert not (directory / "link.log").exists()
    assert not (directory / "observation.json").exists()


@pytest.mark.parametrize("output_name", ["object_path", "output"])
def test_existing_outputs_are_never_overwritten(application, output_name):
    observation, arguments, _receipt = application
    arguments[output_name].symlink_to(arguments["directory"] / "missing")
    with pytest.raises(ValueError, match="already exists"):
        build_driver_link(observation, **arguments)
