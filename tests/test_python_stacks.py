"""Subprocess controls for the diagnostic runner; no compiler or GPU work."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, body, *, arguments=(), interval="0.01", digest=None, output=None,
           setup="", after=""):
    script = tmp_path / "original script.py"
    script.write_text(body)
    output = output or tmp_path / "stacks.jsonl"
    prefix = [sys.executable, "-m", "elmm_pipeline.campaigns.python_stacks"]
    if setup or after:
        prefix = [sys.executable, "-c", "from elmm_pipeline.campaigns import python_stacks as runner\n"
                  + setup + "\nrunner.main()\n" + after]
    command = [*prefix, "--script", str(script), "--script-sha256",
               digest if digest is not None else hashlib.sha256(script.read_bytes()).hexdigest(),
               "--output", str(output), "--interval=" + interval, "--", *arguments]
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONHOME", None)
    result = subprocess.run(command, cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=10, check=False)
    return result, script, output


def records(output):
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_original_script_receives_arguments_and_imports_sibling_once(tmp_path):
    (tmp_path / "sibling.py").write_text("VALUE = 37\n")
    result, script, output = launch(tmp_path, '''import json,sys
from pathlib import Path
import sibling
import __main__
assert __main__.__dict__ is globals()
ordinary_annotation: int = 1
with Path("visits").open("a") as stream:
    stream.write("once\\n")
Path("seen.json").write_text(json.dumps({"argv":sys.argv,"name":__name__,"value":sibling.VALUE,"path0":sys.path[0],"file":__file__,"package":__package__,"spec":__spec__,"loader":__loader__,"cached":__cached__,"annotations_original":__annotations__["ordinary_annotation"] is int}))
''', arguments=("argument with spaces", "--original-option", "last"), interval="60")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "visits").read_text() == "once\n"
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen == {"argv": [str(script), "argument with spaces", "--original-option", "last"],
                    "name": "__main__", "value": 37, "path0": str(tmp_path),
                    "file": str(script), "package": "", "spec": None, "loader": None,
                    "cached": None, "annotations_original": True}
    assert records(output)[0]["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()


def test_periodic_samples_are_bounded_and_never_include_source_or_locals(tmp_path):
    result, script, output = launch(tmp_path, '''import time
def wait_here(depth):
    private_value = "PRIVATE_LOCAL_MUST_NOT_APPEAR"
    if depth:
        return wait_here(depth-1)
    time.sleep(.2)
wait_here(150)
''')
    assert result.returncode == 0, result.stderr
    samples = [row for row in records(output) if row["event"] == "python_stack"]
    assert samples
    for sample in samples:
        assert 0 < len(sample["frames"]) <= 100
        assert type(sample["truncated"]) is bool
        assert all(set(frame) == {"filename", "line", "function"} for frame in sample["frames"])
    assert any(frame["filename"] == str(script) and frame["function"] == "wait_here"
               for sample in samples for frame in sample["frames"])
    assert any(sample["truncated"] and len(sample["frames"]) == 100 for sample in samples)
    assert "PRIVATE_LOCAL_MUST_NOT_APPEAR" not in output.read_text()
    assert "private_value" not in output.read_text()


@pytest.mark.parametrize("ending,code,message", [
    ("raise RuntimeError('original failure')", 1, "RuntimeError: original failure"),
    ("raise SystemExit(7)", 7, ""),
    ("raise SystemExit(0)", 0, ""),
])
def test_original_failure_or_exit_stops_waiting_sampler_without_replay(tmp_path, ending, code, message):
    result, _, output = launch(tmp_path, '''import atexit,json,threading
from pathlib import Path
atexit.register(lambda: Path("threads.json").write_text(json.dumps([thread.name for thread in threading.enumerate()])))
with Path("visits").open("a") as stream:
    stream.write("once\\n")
''' + ending + "\n", interval="60")
    assert result.returncode == code
    assert message in result.stderr
    assert (tmp_path / "visits").read_text() == "once\n"
    assert [row["event"] for row in records(output)] == ["sampler_started"]
    assert "python-stack-sampler" not in json.loads((tmp_path / "threads.json").read_text())


@pytest.mark.parametrize("ending,code", [("", 1), ("raise SystemExit(0)", 1),
                                          ("raise SystemExit(7)", 7),
                                          ("raise RuntimeError('original failure wins')", 1)])
def test_sampler_failure_is_visible_without_masking_original_failure(tmp_path, ending, code):
    result, _, _ = launch(tmp_path, '''import sys,time
from pathlib import Path
def broken_frames():
    raise RuntimeError("injected sampler failure")
sys._current_frames = broken_frames
time.sleep(.1)
with Path("visits").open("a") as stream:
    stream.write("once\\n")
''' + ending + "\n")
    assert result.returncode == code
    assert "Python stack sampler failed" in result.stderr
    assert "injected sampler failure" in result.stderr
    assert (tmp_path / "visits").read_text() == "once\n"
    if "original failure wins" in ending:
        assert "RuntimeError: original failure wins" in result.stderr


def test_log_limit_stops_sampling_without_replaying_script(tmp_path):
    result, _, output = launch(tmp_path, '''import time
from pathlib import Path
time.sleep(.1)
with Path("visits").open("a") as stream:
    stream.write("once\\n")
''', setup="runner.MAX_LOG_BYTES = 1024")
    assert result.returncode != 0
    assert "Python stack log byte limit exceeded" in result.stderr
    assert output.stat().st_size <= 1024
    assert records(output)
    assert (tmp_path / "visits").read_text() == "once\n"


def test_execution_uses_authenticated_bytes_and_restores_runner_namespace(tmp_path):
    setup = '''import sys
from pathlib import Path
saved_main, saved_argv, saved_path = sys.modules["__main__"], sys.argv, sys.path
original_compile = compile
def replace_after_compile(source, filename, mode, **options):
    result = original_compile(source, filename, mode, **options)
    Path(filename).write_text("from pathlib import Path\\nPath('replacement-executed').touch()\\n")
    return result
runner.compile = replace_after_compile
'''
    after = '''assert sys.modules["__main__"] is saved_main
assert sys.argv is saved_argv
assert sys.path is saved_path
'''
    result, script, output = launch(tmp_path, '''from pathlib import Path
with Path("visits").open("a") as stream:
    stream.write("original once\\n")
''', interval="60", setup=setup, after=after)
    assert result.returncode == 0, result.stderr
    assert "replacement-executed" in script.read_text()
    assert records(output)[0]["script_sha256"] != hashlib.sha256(script.read_bytes()).hexdigest()
    assert (tmp_path / "visits").read_text() == "original once\n"
    assert not (tmp_path / "replacement-executed").exists()


@pytest.mark.parametrize("ending,code", [("", 1), ("raise SystemExit(0)", 1),
                                          ("raise SystemExit(7)", 7),
                                          ("raise RuntimeError('original failure wins')", 1)])
def test_close_failure_preserves_original_failure_and_never_replays(tmp_path, ending, code):
    setup = '''from pathlib import Path
class BrokenClose:
    def __init__(self, stream):
        self.stream = stream
    def __getattr__(self, name):
        return getattr(self.stream, name)
    def close(self):
        self.stream.close()
        raise OSError("injected log close failure")
class OutputPath(type(Path())):
    def open(self, mode="r", **options):
        stream = super().open(mode, **options)
        return BrokenClose(stream) if mode == "x" else stream
runner.Path = OutputPath
'''
    result, _, output = launch(tmp_path, '''from pathlib import Path
with Path("visits").open("a") as stream:
    stream.write("once\\n")
''' + ending + "\n", interval="60", setup=setup)
    assert result.returncode == code
    assert "injected log close failure" in result.stderr
    assert (tmp_path / "visits").read_text() == "once\n"
    assert records(output)
    if "original failure wins" in ending:
        assert "RuntimeError: original failure wins" in result.stderr


@pytest.mark.parametrize("digest", ["0" * 64, "not-a-sha256"])
def test_bad_hash_never_executes_or_creates_output(tmp_path, digest):
    result, _, output = launch(tmp_path, 'from pathlib import Path\nPath("executed").touch()\n', digest=digest)
    assert result.returncode != 0
    assert "SHA-256" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "executed").exists()


def test_existing_output_remains_unchanged_and_script_does_not_execute(tmp_path):
    output = tmp_path / "existing.jsonl"
    output.write_text("preserve previous evidence\n")
    result, _, _ = launch(tmp_path, 'from pathlib import Path\nPath("executed").touch()\n', output=output)
    assert result.returncode != 0
    assert output.read_text() == "preserve previous evidence\n"
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "-inf", "invalid"])
def test_invalid_interval_never_executes_or_creates_output(tmp_path, interval):
    result, _, output = launch(tmp_path, 'from pathlib import Path\nPath("executed").touch()\n', interval=interval)
    assert result.returncode != 0
    assert "finite positive" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "executed").exists()
