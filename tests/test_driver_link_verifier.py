"""Real build/relink receipts checked by the independently selected compiler.

The tiny host-only CUDA object tests build provenance, not GPU offloading or
application numerical qualification. No compiler verification is mocked.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from elmm_pipeline.application_build import attest, verify
from elmm_pipeline.build_receipts import CommandRecorder, file_identity, summarize_build
from elmm_pipeline.campaigns.driver_link import build_driver_link


def saved(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return file_identity(path)


def written(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def public_api(root, directory, code, payload):
    directory.mkdir(parents=True, exist_ok=True)
    request, output = directory / "request.json", directory / "response.json"
    saved(request, payload)
    script = "import json,sys\nfrom pathlib import Path\np=json.loads(Path(sys.argv[1]).read_text())\n" + code
    script += "\nPath(sys.argv[2]).write_text(json.dumps(result))\n"
    result = subprocess.run([sys.executable, "-c", script, str(request), str(output)], cwd=root,
                            env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True,
                            check=False, timeout=240)
    saved(directory / "command.json", {"argv": result.args, "returncode": result.returncode, "timeout_seconds": 240})
    (directory / "compiler.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(output.read_text())


def test_executed_diagnostic_is_accepted_by_public_compiler_verifier(tmp_path):
    root = Path(os.environ.get("ELMM_COMPILER_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
    fortran, cuda = shutil.which("gfortran"), shutil.which("nvcc")
    if not (root / "compiler/offload/application_build.py").is_file() or fortran is None or cuda is None:
        pytest.skip("Selected compiler checkout, GNU Fortran and CUDA compiler required")
    cudart = (Path(cuda).resolve().parents[1] / "lib64/libcudart.so").resolve(strict=True)
    original = tmp_path / "immutable"
    model = written(original / "model.f90", "subroutine advance(x)\ninteger::x\nx=x+1\nend\n")
    driver = written(original / "driver.f90", "program driver\ninteger::x=1\ncall advance(x)\nprint '(I0)',x\nend\n")
    old = tmp_path / "origin-artifacts"
    interface = written(old / "interface.f90", "subroutine unused_interface()\nend\n")
    worker = written(old / "worker.cu", 'extern "C" void unused_worker() {}\n')
    cuda_flags = public_api(root, tmp_path / "cuda-options", "from compiler.numerical_contract import cuda_compile_options\n"
                           "result=cuda_compile_options()", {})

    def compile_build(build_root, destinations, generated, macros=None):
        build_root.mkdir(parents=True)
        cwd = Path(next(iter(destinations.values()))).parent
        recorder = CommandRecorder(build_root / "commands")
        objects = []
        sources = [*map(Path, destinations.values()), *generated]
        for index, source in enumerate(sources):
            obj = build_root / ("unit-" + str(index) + ".o")
            is_cuda = source.suffix == ".cu"
            flags = ["-O0", *cuda_flags] if is_cuda else ["-O0", "-I" + str(build_root), "-J" + str(build_root)]
            if macros and str(source) in macros:
                flags.append("-D" + macros[str(source)])
            argv = [cuda if is_cuda else fortran, *flags, "-c", str(source), "-o", str(obj)]
            with (build_root / ("compile-" + str(index) + ".log")).open("wb") as log:
                assert recorder.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode == 0
            objects.append(str(obj))
        binary = build_root / "program"
        assert recorder.run([fortran, "-O0", *objects, str(cudart), "-Wl,-rpath," + str(cudart.parent),
                             "-o", str(binary)], cwd=cwd, timeout=240).returncode == 0
        receipt = summarize_build(binary, [build_root / "commands"],
                                  {str(source): file_identity(source)["sha256"] for source in sources})
        reference = saved(build_root / "receipts.json", receipt)
        return receipt, reference

    origins = {}
    old_manifest = saved(old / "manifest.json", {"sources": {}, "artifacts_sha256": {
        path.name: file_identity(path)["sha256"] for path in (interface, worker)}})
    for mode in ("native", "resident"):
        destinations = {str(path): str(written(tmp_path / (mode + "-source") / path.name, path.read_text()))
                        for path in (model, driver)}
        _receipt, reference = compile_build(tmp_path / (mode + "-origin"), destinations,
                                            [interface, worker] if mode == "resident" else [])
        origins[mode] = {"receipt": reference["path"], "receipt_sha256": reference["sha256"],
                         "build_root": str(tmp_path / (mode + "-origin")),
                         "source_mapping": {destination: source for source, destination in destinations.items()}}
        if mode == "resident":
            origins[mode].update(manifest=old_manifest["path"], manifest_sha256=old_manifest["sha256"])
    bindings = {"sources": {str(model): file_identity(model)["sha256"]},
                "source_inputs_identity": "1" * 64, "configuration_identity": "2" * 64,
                "compiler_identity": "3" * 64, "options_identity": "4" * 64,
                "worker_artifacts": {"worker.cu": file_identity(worker)["sha256"]}, "runtime_identity": "5" * 64}
    contract = public_api(root, tmp_path / "issue-contract",
                          "from compiler.offload.application_build import require_application_build_contract\n"
                          "result=require_application_build_contract(p['requested'],**p['bindings'])",
                          {"requested": {"schema_version": 1, "protocol": "executed-application-build-v1", "origins": origins},
                           "bindings": bindings})
    selection = {"protocol": "checked-interval-process-v4", "sources": bindings["sources"], "build_contract": contract}
    selection_digest = public_api(root, tmp_path / "selection-digest",
                                  "from compiler.offload.application_build import digest\nresult=digest(p)", selection)
    for mode in ("native", "resident"):
        artifacts = tmp_path / (mode + "-artifacts")
        new_interface = written(artifacts / "interface.f90", interface.read_text())
        new_worker = written(artifacts / "worker.cu", worker.read_text())
        selector = written(artifacts / "selector.cu", (
            "#ifndef FORT_EXACT_WINDOW_BUILD_" + selection_digest + "\n#error missing attestation\n#endif\n"
            if mode == "resident" else "") + 'extern "C" void observer() {}\n')
        generated = [new_interface, new_worker, selector]
        manifest = {"source_inputs": bindings["sources"], "runtime": {"runtime_id": bindings["runtime_identity"]},
                    "sources": {}, "scopes": [{"exact_window": {"contract": selection, "contract_sha256": selection_digest,
                                                                  "capture": True, "capture_variant": mode}}],
                    "artifacts_sha256": {path.name: file_identity(path)["sha256"] for path in generated},
                    "build_sources": [{"path": path.name, "language": "cuda" if path.suffix == ".cu" else "fortran",
                                       "role": "shared_entry"} for path in generated]}
        manifest_ref = saved(artifacts / "manifest.json", manifest)
        destinations = {str(path): str(written(tmp_path / (mode + "-capture-source") / path.name, path.read_text()))
                        for path in (model, driver)}
        build_root = tmp_path / (mode + "-capture-build")
        transport = {"compiler_root": str(root), "python": sys.executable}
        request = {"manifest": manifest_ref["path"], "manifest_sha256": manifest_ref["sha256"],
                   "artifact_root": str(artifacts), "build_root": str(build_root), "destinations": destinations}
        token = verify(transport, "preflight", request, tmp_path / (mode + "-preflight"))
        _receipt, reference = compile_build(build_root, destinations, generated, token["macros"])
        application = attest(transport, token, reference["path"], build_root)
        diagnostic_source = written(tmp_path / (mode + "-diagnostic.f90"),
                                    driver.read_text().replace("print '(I0)',x", "print '(A,I0)','diagnostic=',x"))
        diagnostic = build_driver_link(application, original_source=driver, source=diagnostic_source,
                                        object_path=tmp_path / (mode + "-diagnostic.o"),
                                        output=tmp_path / (mode + "-diagnostic"), directory=tmp_path / (mode + "-relink"))
        approved_driver = {"original_source": str(driver), "original_sha256": file_identity(driver)["sha256"],
                           "replacement_sha256": file_identity(diagnostic_source)["sha256"]}
        payload = {"diagnostic": diagnostic, "application": application, "contract": contract,
                   "driver": approved_driver, "mode": mode}
        verified = public_api(root, tmp_path / (mode + "-verify-relink"),
                              "from compiler.offload.application_build import application_build_validation,validate_diagnostic_relink\n"
                              "with application_build_validation():\n"
                              " result=validate_diagnostic_relink(p['diagnostic'],application_build=p['application'],"
                              "contract=p['contract'],driver=p['driver'],mode=p['mode'])", payload)
        assert verified["binary_sha256"] == file_identity(verified["binary"])["sha256"]
        result = subprocess.run([verified["binary"]], text=True, capture_output=True, check=False, timeout=5)
        assert result.returncode == 0 and result.stdout.strip() == "diagnostic=2"
