"""Independent CUDA compilation shares the host build's bounded jobs budget."""

import hashlib
import json
import subprocess
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import pytest

from elmm_pipeline.build import build
from elmm_pipeline.gpu_toolchain import compile_gpu_sources


def sources(tmp_path, languages):
    units = []
    for index, language in enumerate(languages):
        directory = tmp_path / f"source {index}"
        directory.mkdir()
        # Repeated stems must still get separate object and diagnostic paths.
        source = directory / ("unit.cu" if language == "cuda" else "unit.f90")
        source.write_text(f"source {index}\n")
        units.append({"source": source, "language": language})
    return units


def toolchain():
    return {"architecture": "90", "nvcc": "/mock/nvcc", "cuda_host_cxx": "/mock/c++",
            "cuda_library_dir": "/mock/cuda/lib"}


def position(command):
    return int(Path(command[command.index("-o") + 1]).name.split("_", 1)[0])


@pytest.mark.parametrize("jobs", [1, 2, 8])
def test_cuda_jobs_are_bounded_and_inventory_and_logs_remain_source_ordered(tmp_path, jobs):
    units = sources(tmp_path, ["cuda"] * 4)
    workers = min(jobs, len(units))
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    first_finished = threading.Event()
    active = peak = 0
    completed = []

    def compiled(command, **kwargs):
        nonlocal active, peak
        index = position(command)
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        if workers > 1 and index == 0:
            assert first_finished.wait(5)
        kwargs["stdout"].write(f"diagnostics {index}\n")
        Path(command[command.index("-o") + 1]).touch()
        with lock:
            completed.append(index)
            active -= 1
        if index == 1:
            first_finished.set()
        return subprocess.CompletedProcess(command, 0)

    def runtime_query(*args, **kwargs):
        assert active == 0
        assert len(completed) == len(units)
        return "libstdc++.so"

    with patch("elmm_pipeline.gpu_toolchain.subprocess.run", side_effect=compiled), \
            patch("elmm_pipeline.gpu_toolchain.subprocess.check_output", side_effect=runtime_query):
        result = compile_gpu_sources(units, tmp_path / "build", toolchain=toolchain(), jobs=jobs)
    assert peak == workers
    if workers > 1:
        assert completed.index(1) < completed.index(0)
    assert result["jobs"] == jobs
    assert [position(command) for command in result["commands"]] == list(range(4))
    assert [int(Path(path).name[:3]) for path in result["objects"]] == list(range(4))
    assert len({unit["build_log"] for unit in result["sources"]}) == 4
    for original, compiled_unit in zip(units, result["sources"], strict=True):
        assert compiled_unit["build_status"] == "compiled"
        assert compiled_unit["sha256"] == hashlib.sha256(original["source"].read_bytes()).hexdigest()
    log = (tmp_path / "build/gpu/gpu-build.log").read_text()
    assert [log.index(f"diagnostics {index}") for index in range(4)] == sorted(
        log.index(f"diagnostics {index}") for index in range(4))
    assert json.loads((tmp_path / "build/gpu/gpu-build-command.json").read_text()) == result


def test_failed_cuda_cancels_pending_and_joins_running_before_raising(tmp_path):
    units = sources(tmp_path, ["cuda"] * 12)
    second_started = threading.Event()
    release_running = threading.Event()
    cancelled_pending = threading.Event()
    lock = threading.Lock()
    active = 0
    failures = []
    original_cancel = Future.cancel

    def cancel(future):
        result = original_cancel(future)
        if result:
            cancelled_pending.set()
        return result

    def compiled(command, **kwargs):
        nonlocal active
        index = position(command)
        with lock:
            active += 1
        try:
            kwargs["stdout"].write(f"diagnostics {index}\n")
            if index == 0:
                assert second_started.wait(5)
                return subprocess.CompletedProcess(command, 7)
            second_started.set()
            assert release_running.wait(5)
            Path(command[command.index("-o") + 1]).touch()
            return subprocess.CompletedProcess(command, 0)
        finally:
            with lock:
                active -= 1

    def invoke():
        try:
            compile_gpu_sources(units, tmp_path / "build", toolchain=toolchain(), jobs=2)
        except RuntimeError as error:
            failures.append(error)

    with patch("elmm_pipeline.gpu_toolchain.subprocess.run", side_effect=compiled), \
            patch("elmm_pipeline.gpu_toolchain.subprocess.check_output") as runtime, \
            patch.object(Future, "cancel", cancel):
        caller = threading.Thread(target=invoke)
        caller.start()
        try:
            assert cancelled_pending.wait(5)
            assert caller.is_alive()
            assert active > 0
            runtime.assert_not_called()
        finally:
            release_running.set()
            caller.join(timeout=5)
        assert not caller.is_alive()
        assert active == 0
        runtime.assert_not_called()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    saved = json.loads((tmp_path / "build/gpu/gpu-build-command.json").read_text())
    assert saved["sources"][0]["build_status"] == "failed"
    assert "compiler exit 7" in saved["sources"][0]["build_error"]
    assert "cancelled" in {unit["build_status"] for unit in saved["sources"]}
    assert len(saved["commands"]) == len(units)
    assert all(Path(path).is_file() for path in saved["objects"])
    log = (tmp_path / "build/gpu/gpu-build.log").read_text()
    assert "diagnostics 0" in log
    assert "cancelled" in log


@pytest.mark.parametrize("failure", ["exception", "missing_object"])
def test_failed_unit_retains_diagnostics_without_runtime_discovery(tmp_path, failure):
    units = sources(tmp_path, ["cuda"])

    def compiled(command, **kwargs):
        kwargs["stdout"].write("compiler diagnostic\n")
        if failure == "exception":
            raise OSError("compiler could not start")
        return subprocess.CompletedProcess(command, 0)

    with patch("elmm_pipeline.gpu_toolchain.subprocess.run", side_effect=compiled), \
            patch("elmm_pipeline.gpu_toolchain.subprocess.check_output") as runtime, \
            pytest.raises(RuntimeError, match="GPU compilation failed"):
        compile_gpu_sources(units, tmp_path / "build", toolchain=toolchain(), jobs=2)
    runtime.assert_not_called()
    saved = json.loads((tmp_path / "build/gpu/gpu-build-command.json").read_text())
    assert saved["sources"][0]["build_status"] == "failed"
    assert "compiler diagnostic" in Path(saved["sources"][0]["build_log"]).read_text()
    assert saved["objects"] == []




@pytest.mark.parametrize("jobs", [0, -1, True, 1.5])
def test_invalid_jobs_reject_before_any_toolchain_work(jobs, tmp_path):
    with patch("elmm_pipeline.gpu_toolchain.discover_toolchain") as discover, \
            pytest.raises(ValueError, match="jobs must be a positive integer"):
        compile_gpu_sources([], tmp_path, jobs=jobs)
    discover.assert_not_called()


def test_build_forwards_jobs_and_gpu_failure_prevents_scons(tmp_path):
    elmm, poisfft = tmp_path / "ELMM", tmp_path / "PoisFFT"
    (elmm / "src").mkdir(parents=True)
    (poisfft / "src").mkdir(parents=True)
    (poisfft / "src/poisfft.f90").touch()
    (elmm / "src/SConstruct").write_text("t = env.Program()\n")
    units = sources(tmp_path, ["cuda"])
    with patch("elmm_pipeline.build.compile_gpu_sources", side_effect=RuntimeError("unit failed")) as compile_gpu, \
            patch("elmm_pipeline.build.subprocess.run") as scons, \
            pytest.raises(RuntimeError, match="unit failed"):
        build(elmm, poisfft, tmp_path / "build", Path("python"), tmp_path / "lib", 6, 4, [], gpu_sources=units)
    assert compile_gpu.call_args.kwargs["jobs"] == 6
    assert compile_gpu.call_args.kwargs["defines"] == ["DPREC"]
    scons.assert_not_called()
    assert not (elmm / "src/SConstruct.pipeline").exists()
