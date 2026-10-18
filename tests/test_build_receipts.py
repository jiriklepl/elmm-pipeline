"""Actual argv receipts remain independent of diagnostic text and shell parsing."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from elmm_pipeline.build_receipts import (
    CommandRecorder,
    read_receipts,
    require_build_receipts,
    snapshot_inputs,
    summarize_build,
)


@pytest.fixture
def compiler(tmp_path):
    path = tmp_path / "compiler with spaces"
    path.write_text("#!" + sys.executable + "\n" + r'''
import json, os, pathlib, sys
if '--version' in sys.argv:
    print('receipt-test-compiler 1')
    sys.exit(0)
args = sys.argv[1:]
if '--remove' in args:
    pathlib.Path(args[args.index('--remove')+1]).unlink()
if '--fail' in args:
    print('gfortran -O3 -c invented.f90 -o invented.o')
    sys.exit(7)
if '--mutate' in args:
    pathlib.Path(args[args.index('--mutate')+1]).write_text('changed')
target = pathlib.Path(args[args.index('-o')+1])
target.write_text(json.dumps({'argv': args, 'sentinel': os.environ.get('FCFLAGS')}))
''')
    path.chmod(0o755)
    return path


def receipts(directory):
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_argv_order_inputs_headers_outputs_and_environment(compiler, tmp_path):
    source = tmp_path / "source file.f90"
    source.write_text("end\n")
    header = tmp_path / "values.inc"
    header.write_text("! configured constants\n")
    output = tmp_path / "object file.o"
    directory = tmp_path / "receipts"
    argv = [str(compiler), "-O3", "-O1", "-DVALUE=a b", "-c", str(source), "-o", str(output)]
    environment = {**os.environ, "FCFLAGS": "-fprotect-parens", "PRIVATE_CREDENTIAL": "not-for-reporting"}
    assert CommandRecorder(directory).run(argv, cwd=tmp_path, env=environment,
                                          additional_inputs=[header]).returncode == 0
    [record] = receipts(directory)
    assert record["argv"] == argv
    assert json.loads(output.read_text())["argv"] == argv[1:]
    assert record["environment"]["FCFLAGS"] == "-fprotect-parens"
    assert "PRIVATE_CREDENTIAL" not in record["environment"]
    assert "not-for-reporting" not in json.dumps(record)
    assert len(record["environment_sha256"]) == 64
    assert record["compiler"] == {"path": str(compiler), "sha256": digest(compiler),
                                  "version": "receipt-test-compiler 1", "version_returncode": 0}
    assert {item["path"]: item["sha256"] for item in record["inputs"]} == {
        str(source): digest(source), str(header): digest(header)}
    assert record["outputs"] == [{"path": str(output), "sha256": digest(output)}]
    assert record["status"] == "completed" and record["returncode"] == 0
    assert record["shell"] is False
    assert read_receipts(directory)[0]["execution"] == record


def test_scons_quoted_arguments_are_executed_without_expansion(compiler, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sentinel = tmp_path / "injected"
    output = tmp_path / "object file.o"
    argv = [str(compiler), "-DVALUE=$(touch " + str(sentinel) + ")", "-o", str(output)]
    recorder = CommandRecorder(tmp_path / "receipts")
    assert recorder.scons_spawn("/bin/sh", None, argv[0], [shlex.quote(arg) for arg in argv], os.environ) == 0
    assert not sentinel.exists()
    assert json.loads(output.read_text())["argv"] == argv[1:]
    assert receipts(tmp_path / "receipts")[0]["argv"] == argv


def test_relative_executable_resolves_in_command_directory(compiler, tmp_path):
    output = tmp_path / "object.o"
    argv = ["./" + compiler.name, "-o", str(output)]
    assert CommandRecorder(tmp_path / "receipts").run(argv, cwd=tmp_path).returncode == 0
    assert receipts(tmp_path / "receipts")[0]["compiler"]["path"] == str(compiler)


def test_failure_diagnostic_is_not_a_build_command(compiler, tmp_path):
    directory = tmp_path / "receipts"
    result = CommandRecorder(directory).run([str(compiler), "--fail"], cwd=tmp_path,
                                             stdout=subprocess.DEVNULL)
    assert result.returncode == 7
    [record] = receipts(directory)
    assert record["returncode"] == 7 and record["outputs"] == []
    assert record["argv"] == [str(compiler), "--fail"]
    assert "invented" not in json.dumps(record)
    with pytest.raises(ValueError, match="successful"):
        read_receipts(directory)


def test_input_mutation_cannot_appear_as_unchanged_success(compiler, tmp_path):
    source = tmp_path / "source.f90"
    source.write_text("before")
    before = digest(source)
    directory = tmp_path / "receipts"
    result = CommandRecorder(directory).run([str(compiler), "--mutate", str(source), "-o", "object.o"], cwd=tmp_path)
    assert result.returncode == 0
    [record] = receipts(directory)
    assert record["status"] == "inputs_changed"
    assert record["inputs"] == [{"path": str(source), "sha256": before}]
    with pytest.raises(ValueError, match="successful"):
        read_receipts(directory)


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("removed", ["module", "compiler"])
def test_disappearing_build_input_preserves_actual_exit_status(compiler, tmp_path, failed, removed):
    module = tmp_path / "cached.mod"
    module.write_text("old module")
    target = module if removed == "module" else compiler
    directory = tmp_path / "receipts"
    argv = [str(compiler), "--remove", str(target), "-o", "object.o"]
    if failed:
        argv.append("--fail")
    result = CommandRecorder(directory).run(argv, cwd=tmp_path, additional_inputs=[module],
                                             stdout=subprocess.DEVNULL)
    assert result.returncode == (7 if failed else 0)
    [record] = receipts(directory)
    assert record["returncode"] == result.returncode
    assert record["status"] == "inputs_changed"
    assert "error" not in record
    assert not target.exists()
    with pytest.raises(ValueError, match="successful"):
        read_receipts(directory)


@pytest.mark.parametrize("changed", ["source", "header", "object", "compiler"])
def test_reader_rejects_stale_executed_receipts(compiler, tmp_path, changed):
    source, header, output = (tmp_path / name for name in ("source.f90", "config.inc", "object.o"))
    source.write_text("source")
    header.write_text("header")
    directory = tmp_path / "receipts"
    CommandRecorder(directory).run([str(compiler), "-c", str(source), "-o", str(output)],
                                    additional_inputs=[header])
    assert len(read_receipts(directory)) == 1
    target = {"source": source, "header": header, "object": output, "compiler": compiler}[changed]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file changed"):
        read_receipts(directory)


def test_parallel_commands_retain_every_distinct_object_and_flag(compiler, tmp_path):
    directory = tmp_path / "receipts"
    recorder = CommandRecorder(directory)
    commands = [[str(compiler), "-DINDEX=" + str(index), "-o", "object-" + str(index) + ".o"] for index in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda argv: recorder.run(argv, cwd=tmp_path), commands))
    assert all(result.returncode == 0 for result in results)
    records = receipts(directory)
    assert len(records) == 8
    assert {tuple(record["argv"]) for record in records} == {tuple(command) for command in commands}
    assert all(record["status"] == "completed" for record in records)
    assert not list(directory.glob("*.tmp"))


@pytest.fixture
def built_receipts(compiler, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.f90").write_text("end\n")
    (source / "model.inc").write_text("! configured values\n")
    inputs = snapshot_inputs([source])
    directory = tmp_path / "receipts"
    recorder = CommandRecorder(directory)
    obj, binary = tmp_path / "model.o", tmp_path / "program"
    recorder.run([str(compiler), "-O1", "-c", str(source / "model.f90"), "-o", str(obj)])
    recorder.run([str(compiler), str(obj), "-o", str(binary)])
    return source, directory, obj, binary, inputs


def test_summary_requires_actual_commands_and_binds_source_headers_and_receipts(built_receipts, tmp_path):
    source, directory, obj, binary, inputs = built_receipts
    summary = summarize_build(binary, [directory], inputs)
    assert summary["status"] == "complete" and summary["placement_authority"] is False
    assert summary["units"][0]["object"] == str(obj)
    assert "-O1" in summary["units"][0]["execution"]["argv"]
    assert str(source / "model.inc") in summary["files_sha256"]
    path = tmp_path / "build-receipts.json"
    path.write_text(json.dumps(summary))
    assert require_build_receipts(path) == summary
    receipt = next(directory.glob("*.json"))
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(ValueError, match="file changed"):
        require_build_receipts(path)


def test_summary_cannot_reconstruct_missing_compile_receipts_from_outputs(built_receipts):
    _source, directory, _obj, binary, inputs = built_receipts
    for receipt in directory.glob("*.json"):
        if "-c" in json.loads(receipt.read_text())["argv"]:
            receipt.unlink()
    with pytest.raises(ValueError, match="lack linked object command"):
        summarize_build(binary, [directory], inputs)


def test_prebuild_header_changes_reject_summary(built_receipts):
    source, directory, _obj, binary, inputs = built_receipts
    (source / "model.inc").write_text("changed configuration")
    with pytest.raises(ValueError, match="source/header changed"):
        summarize_build(binary, [directory], inputs)


def test_summary_rejects_edited_flags_without_file_changes(built_receipts, tmp_path):
    _source, directory, _obj, binary, inputs = built_receipts
    summary = summarize_build(binary, [directory], inputs)
    summary["units"][0]["execution"]["argv"][1] = "-Ofast"
    path = tmp_path / "build-receipts.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="identity differs"):
        require_build_receipts(path)


def test_real_scons_preserves_per_source_flags_and_builds_with_spaces(tmp_path):
    python = Path(os.environ.get("ELMM_SCONS_PYTHON", str(Path(__file__).resolve().parents[1] / ".cache/envs/build/bin/python")))
    if not python.is_file() or not shutil.which("gfortran"):
        pytest.skip("Requires the pipeline SCons environment and GNU Fortran")
    directory = tmp_path / "source tree"
    directory.mkdir()
    (directory / "one.f90").write_text("subroutine one(x)\ninteger::x\nx=x+1\nend\n")
    (directory / "main.f90").write_text("program main\ninteger::x=1\ncall one(x)\nif(x/=2)stop 1\nend\n")
    module = Path(__file__).resolve().parents[1] / "elmm_pipeline/build_receipts.py"
    (directory / "SConstruct").write_text(f"""import importlib.util, os
spec=importlib.util.spec_from_file_location('receipts', {str(module)!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
recorder=module.CommandRecorder({str(directory / 'receipts')!r})
env=Environment(ENV=os.environ, FORTRAN='gfortran', F90='gfortran', LINK='gfortran')
objects=env.Object('one.f90', F90FLAGS=['-O1', '-fprotect-parens'])
objects+=env.Object('main.f90', F90FLAGS=['-O3', '-ffp-contract=off'])
env.Replace(SPAWN=recorder.scons_spawn)
Default(env.Program(target='program with spaces', source=objects))
""")
    result = subprocess.run([str(python), "-m", "SCons", "-j", "2"], cwd=directory,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    binary = directory / "program with spaces"
    assert subprocess.run([str(binary)], check=False).returncode == 0
    records = receipts(directory / "receipts")
    assert len(records) == 3
    commands = {Path(record["outputs"][0]["path"]).name: record["argv"] for record in records}
    assert "-O1" in commands["one.o"] and "-O3" not in commands["one.o"]
    assert "-O3" in commands["main.o"] and "-O1" not in commands["main.o"]
    assert all(record["status"] == "completed" for record in records)
