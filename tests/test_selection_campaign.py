"""Real process measurement and explicitly synthetic campaign transport tests."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from elmm_pipeline import selection_campaign as collector
from elmm_pipeline.build_receipts import (
    CommandRecorder,
    require_build_receipts,
    summarize_build,
)


def policy():
    return {"schema_version": 2, "identity": "policy", "fields": {
        "state": {"payload": "fields.bin", "metadata": "fields.shape"}},
        "outputs": {"format": "raw-float64-storage-v1", "files": {
            name: {"payload": name + ".bin", "metadata": name + ".shape"} for name in ("final", "average")}}}


@pytest.fixture
def real_program(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required")
    source, obj, binary = (tmp_path / name for name in ("program.c", "program.o", "program"))
    source.write_text(r'''
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <time.h>
int main(int argc, char **argv) {
 if(argc>1 && !strcmp(argv[1],"timeout")) {struct timespec t={1,0};nanosleep(&t,0);}
 if(argc>1 && !strcmp(argv[1],"fail")) return 7;
 if(argc>1 && !strcmp(argv[1],"mutate")) {FILE *f=fopen("case.conf","w");fputs("changed",f);fclose(f);}
 const char *names[]={"final","average","fields"}; double values[]={1.,2.,3.,4.};
 for(int i=0;i<3;++i) {
  char path[80];snprintf(path,sizeof(path),"%s.bin",names[i]);FILE *f=fopen(path,"wb");
  if(!f||fwrite(values,sizeof(double),4,f)!=4)return 8;fclose(f);
  snprintf(path,sizeof(path),"%s.shape",names[i]);f=fopen(path,"w");
  if(!f)return 9;fputs("64\n4\n-1\n2\n",f);fclose(f);
 }
 puts("FORT_APPLICATION_COMPLETION {\"steps\":3,\"natural_end\":true}");
 return 0;
}
''')
    receipts = tmp_path / "commands"
    recorder = CommandRecorder(receipts)
    assert recorder.run([compiler, "-O0", "-c", str(source), "-o", str(obj)], cwd=tmp_path).returncode == 0
    assert recorder.run([compiler, str(obj), "-o", str(binary)], cwd=tmp_path).returncode == 0
    build = summarize_build(binary, [receipts], {str(source): collector._hash(source)})
    receipt = tmp_path / "build.json"
    collector._write(receipt, build)
    assert require_build_receipts(receipt) == build
    config = tmp_path / "configuration"
    config.mkdir()
    (config / "case.conf").write_text("unchanged workload\n")
    workload = {"argv": ["application"], "environment": {"TEST_APPLICATION_INPUT": "fixed"},
                "files": {"case.conf": collector._hash(config / "case.conf")},
                "completion": {"steps": 3, "natural_end": True}}
    workload["identity"] = collector._digest(workload)
    arguments = {"executable": {"binary": str(binary), "binary_sha256": collector._hash(binary)},
                 "workload": workload, "configuration": config,
                 "environment": {**os.environ, "TEST_APPLICATION_INPUT": "fixed"}, "threads": 4,
                 "timeout": 3, "completion": {"kind": "json-stdout"}, "policy": policy()}
    return arguments, receipt


def test_real_compiled_process_records_actual_outputs_completion_and_build_receipts(tmp_path, real_program):
    arguments, receipt = real_program
    observed = collector._run_process(tmp_path / "run", **arguments)
    record = collector._read(observed["run"])
    assert record["exit_code"] == 0
    assert record["timed_out"] is False
    assert record["completion"] == {"steps": 3, "natural_end": True}
    assert record["diagnostic_complete_process_seconds"] > 0
    assert record["argv"] == [arguments["executable"]["binary"]]
    assert record["output_hashes"] == {name: collector._hash(tmp_path / "run" / name)
                                        for name in ("final.bin", "final.shape", "average.bin", "average.shape")}
    assert require_build_receipts(receipt)["binary"] == record["argv"][0]
    # A successfully measured CPU executable has no invented GPU observations.
    with pytest.raises(ValueError, match="observations.*missing"):
        collector._invocations(observed["observations"], selection="anything", mode="resident")


@pytest.mark.parametrize("mode,reason", [("fail", "successfully"), ("mutate", "input changed"),
                                        ("timeout", "successfully")])
def test_real_process_failure_retains_receipt_and_existing_runtime_limit(tmp_path, real_program, mode, reason):
    arguments, _receipt = real_program
    arguments["workload"]["argv"].append(mode)
    arguments["timeout"] = .02 if mode == "timeout" else 3
    directory = tmp_path / "failed"
    with pytest.raises(ValueError, match=reason):
        collector._run_process(directory, **arguments)
    record = json.loads((directory / "run.json").read_text())
    assert record["application_timeout_seconds"] == arguments["timeout"]
    assert record["timed_out"] is (mode == "timeout")
    assert record["diagnostic_complete_process_seconds"] > 0


def test_completion_is_observed_not_copied_from_expected_workload(tmp_path, real_program):
    arguments, _receipt = real_program
    arguments["workload"]["completion"]["steps"] = 999
    with pytest.raises(ValueError, match="observed application completion differs"):
        collector._run_process(tmp_path / "run", **arguments)
    record = json.loads((tmp_path / "run/run.json").read_text())
    assert record["completion"]["steps"] == 3


def test_public_verifier_transport_has_separate_240_second_cap(tmp_path, monkeypatch):
    root = tmp_path / "compiler"
    (root / "compiler/offload").mkdir(parents=True)
    (root / "compiler/offload/selection_cost.py").touch()

    def invoke(argv, **kwargs):
        assert kwargs["timeout"] == 240
        assert kwargs["cwd"] == root
        assert kwargs["env"]["PYTHONPATH"] == str(root)
        response = {"status": "verified", "placement_authority": False}
        response["identity"] = collector._digest(response)
        collector._write(argv[argv.index("--output") + 1], response)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(collector.subprocess, "run", invoke)
    response = collector._verifier({"compiler_root": str(root), "python": sys.executable}, "builds", {}, tmp_path / "verify")
    assert response["placement_authority"] is False


@pytest.mark.parametrize("streams", [(b"partial output", b"partial error"), (None, None)])
def test_verifier_timeout_retains_command_streams_and_virtualenv_launcher(tmp_path, monkeypatch, streams):
    root = tmp_path / "compiler"
    (root / "compiler/offload").mkdir(parents=True)
    (root / "compiler/offload/selection_cost.py").touch()
    launcher = tmp_path / "venv/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)

    def invoke(argv, **kwargs):
        assert argv[0] == str(launcher)
        assert kwargs["timeout"] == 240
        raise subprocess.TimeoutExpired(argv, 240, *streams)

    monkeypatch.setattr(collector.subprocess, "run", invoke)
    directory = tmp_path / "verify"
    with pytest.raises(ValueError, match="240-second phase limit"):
        collector._verifier({"compiler_root": str(root), "python": str(launcher)}, "builds", {}, directory)
    command = json.loads((directory / "command.json").read_text())
    assert command["timed_out"] is True
    assert command["returncode"] is None
    assert command["timeout_seconds"] == 240
    assert command["argv"][0] == str(launcher)
    assert (directory / "stdout.log").read_text() == ("partial output" if streams[0] else "")
    assert (directory / "stderr.log").read_text() == ("partial error" if streams[1] else "")
    assert not (directory / "response.json").exists()


def test_full_campaign_transport_keeps_warmups_diagnostics_and_all_timing_pairs_separate(tmp_path, monkeypatch):
    """Synthetic observer/build fixture tests wiring; supplies no GPU evidence."""
    (tmp_path / "case.conf").write_text("fixed fixture\n")
    workload = {"argv": ["application"], "environment": {}, "files": {"case.conf": collector._hash(tmp_path / "case.conf")},
                "completion": {"steps": 3}}
    workload["identity"] = collector._digest(workload)
    contract = {"schema_version": 2, "protocol": collector.PROTOCOL,
                "acceptance": {"criterion": "paired_non_regression"}, "threads": 4,
                "workloads": [workload], "selection_identity": "selection", "selection_bundle": {"identity": "bundle"},
                "validation_policy": policy(), "build_contract": {"synthetic_fixture": True}}
    collector._write(tmp_path / "contract.json", contract)
    request = {"schema_version": 1, "contract": collector._reference(tmp_path / "contract.json"),
               "compiler": {}, "builds": {mode: {"fixture_mode": mode} for mode in ("native", "resident")},
               "diagnostics": {mode: {"fixture_dump": mode} for mode in ("native", "resident")},
               "configuration_directory": str(tmp_path), "workload_identity": workload["identity"],
               "application_timeout_seconds": 60, "completion": {"kind": "json-stdout"}}
    identity = {"cpu_name": "fixture", "fortran_version": "fixture", "fortran_options": "-O0", "affinity": [0],
                "effective_schedule": {"kind": 1, "chunk": 0}, "environment": {}, "threads": 4,
                "dynamic": 0, "proc_bind": 0, "thread_limit": 4, "max_active_levels": 1}
    calls = []

    def measured(directory, **arguments):
        assert arguments["timeout"] == 60
        directory.mkdir(parents=True)
        mode = arguments["executable"]["binary"]
        calls.append(str(directory.relative_to(directory.parents[1])))
        record = {"diagnostic_complete_process_seconds": 1.0 + len(calls) / 100,
                  "exit_code": 0, "completion": {"steps": 3}}
        collector._write(directory / "run.json", record)
        observation = {"selection": "selection", "key": [4], "outcome": mode, "complete": True,
                       "fallbacks": 0, "launches": 0 if mode == "native" else 3, "seconds": .001,
                       "cpu_identity": identity if mode == "native" else None,
                       "identity": identity if mode == "resident" else None}
        (directory / "stderr.log").write_text(collector.OBSERVATION + json.dumps(observation) + "\n")
        for field in ("final", "average", "fields"):
            (directory / (field + ".bin")).write_bytes(b"synthetic payload")
            (directory / (field + ".shape")).write_text("synthetic bounds")
        return {"run": collector._reference(directory / "run.json"),
                "observations": collector._reference(directory / "stderr.log"),
                "outputs": {field: {kind: collector._reference(directory / (field + suffix))
                                   for kind, suffix in (("payload", ".bin"), ("metadata", ".shape"))}
                            for field in ("final", "average")}}

    def verifier(_transport, phase, payload, _directory):
        if phase == "builds":
            return {"contract_sha256": collector._digest(contract), "variants": {
                mode: {kind: {"binary": mode} for kind in ("application", "diagnostic")}
                for mode in ("native", "resident")}}
        evidence = collector._read(payload["evidence"])
        campaign, = evidence["campaigns"]
        numerical = collector._read(campaign["numerical_validation"])
        assert len(campaign["observations"]) == 18
        assert [row["mode"] for row in campaign["observations"]] == [
            ("native", "resident")[(round_index + turn) % 2] for round_index in range(9) for turn in range(2)]
        assert [row["role"] for row in campaign["observations"]] == ["fit"] * 14 + ["holdout"] * 4
        assert len(numerical["pairs"]) == 9
        assert all(set(pair[mode]["outputs"]) == {"final", "average"}
                   for pair in numerical["pairs"] for mode in ("native", "resident"))
        assert all(set(variant["fields"]) == {"state"} for variant in numerical["variants"].values())
        return {"decision": {"available": True, "selection": "native"}, "evidence": evidence}

    monkeypatch.setattr(collector, "_run_process", measured)
    monkeypatch.setattr(collector, "_verifier", verifier)
    result = collector.run(request, tmp_path / "campaign")
    assert calls[:4] == ["warmup/native", "warmup/resident", "diagnostic/native", "diagnostic/resident"]
    assert len(calls) == 22
    assert result["status"] == "complete"
    assert result["qualification"] is False
    assert result["placement_authority"] is False
    emitted = collector._read(result["scope_cost"])
    assert emitted["acceptance"] == {"criterion": "paired_non_regression"}
    assert emitted["coverage"] == "checked_interval"
    with pytest.raises(FileExistsError):
        collector.run(request, tmp_path / "campaign")

    def declined(directory, **arguments):
        result = measured(directory, **arguments)
        if arguments["executable"]["binary"] == "resident":
            path = directory / "stderr.log"
            row = json.loads(path.read_text().removeprefix(collector.OBSERVATION))
            row["fallbacks"], row["outcome"] = 1, "fallback"
            path.write_text(collector.OBSERVATION + json.dumps(row) + "\n")
            result["observations"] = collector._reference(path)
        return result

    calls.clear()
    monkeypatch.setattr(collector, "_run_process", declined)
    with pytest.raises(ValueError, match="selected alternative"):
        collector.run(request, tmp_path / "declined")
    assert calls == ["warmup/native", "warmup/resident"]
    assert not (tmp_path / "declined/scope-cost.json").exists()
    assert json.loads((tmp_path / "declined/report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("change", [None, "missing", "duplicate", "contract", "mutated"])
def test_combination_requires_one_contract_and_complete_distinct_workload_coverage(tmp_path, monkeypatch, change):
    """Compiler decisions are opaque in this combination transport fixture."""
    contract = {"workloads": [{"identity": "a"}, {"identity": "b"}], "build_contract": {}, "validation_policy": {},
                "acceptance": {"criterion": "paired_non_regression"}}
    references = []
    for index, workload in enumerate(contract["workloads"]):
        current = json.loads(json.dumps(contract))
        if change == "contract" and index:
            current["build_contract"] = {"different": True}
        campaign = {"contract": current, "workload": workload}
        cost = {"coverage": "checked_interval", "acceptance": contract["acceptance"],
                "evidence": {"protocol": collector.PROTOCOL, "campaigns": [campaign]}}
        path = tmp_path / (workload["identity"] + ".json")
        collector._write(path, cost)
        references.append(collector._reference(path))
    if change == "missing":
        references.pop()
    if change == "duplicate":
        references[1] = references[0]

    def verifier(_transport, phase, payload, _directory):
        assert phase == "evaluate"
        evidence = collector._read(payload["evidence"])
        assert {row["workload"]["identity"] for row in evidence["campaigns"]} == {"a", "b"}
        if change == "mutated":
            (tmp_path / "a.json").write_text("{}")
        return {"evidence": evidence, "decision": {"available": True, "selection": "native"}}

    monkeypatch.setattr(collector, "_verifier", verifier)
    requested = {"schema_version": 1, "compiler": {}, "scope_costs": references}
    if change:
        with pytest.raises(ValueError):
            collector.combine(requested, tmp_path / "combined")
        assert not (tmp_path / "combined/scope-cost.json").exists()
    else:
        result = collector.combine(requested, tmp_path / "combined")
        assert result["measured_workloads"] == 2
        assert result["decision"]["selection"] == "native"
        assert result["placement_authority"] is False
