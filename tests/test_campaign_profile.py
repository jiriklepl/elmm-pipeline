"""Bounded collection must not mistake profiler exit for target completion."""
from pathlib import Path
import sys
import time

import pytest

from elmm_pipeline.campaigns import profile as profiling
from elmm_pipeline.campaigns.identity import read, write


def script(path, body):
    path.write_text('#!' + sys.executable + '\n' + body)
    path.chmod(0o755)
    return path


def fixture(tmp_path, monkeypatch, *, exit_code=0, profiler_mode='early', timeout=2):
    root = tmp_path / 'frozen campaign'
    root.mkdir()
    (root / 'config').mkdir()
    (root / 'campaign.json').write_text('{}\n')
    (root / 'reference.vtk').write_text('reference')
    binary = script(root / 'application', '''from pathlib import Path
import sys
import time
with Path('execution-count').open('a') as count:
    count.write('once\\n')
print('partial output', flush=True)
time.sleep(0.15)
Path('output').mkdir()
Path('output/out.vtk').write_text('complete fields')
print('ALL_STEPS_COMPLETE', flush=True)
sys.exit(''' + str(exit_code) + ')\n')
    # This reproduces a profiler returning successfully before its target exits.
    body = '''import subprocess
import sys
target = next(i+1 for i, arg in enumerate(sys.argv) if arg.startswith('--output='))
'''
    if profiler_mode == 'early':
        body += '''from pathlib import Path
subprocess.Popen(sys.argv[target:])
assert not Path(sys.argv[target+2]).exists()
Path('profiler-returned-before-completion').write_text('yes')
'''
    elif profiler_mode == 'forward':
        body += 'sys.exit(subprocess.run(sys.argv[target:]).returncode)\n'
    elif profiler_mode == 'failure':
        body += 'sys.exit(9)\n'
    elif profiler_mode == 'missing':
        body += 'sys.exit(0)\n'
    elif profiler_mode == 'timeout':
        body += 'import time\ntime.sleep(1)\n'
    elif profiler_mode in ('non-object', 'malformed'):
        value = '[]' if profiler_mode == 'non-object' else '{invalid'
        body += 'from pathlib import Path\nPath(sys.argv[target+2]).write_text(' + repr(value) + ')\n'
    else:
        raise AssertionError(profiler_mode)
    profiler = script(tmp_path / 'fake profiler', body)
    state = {'cases': [{'name': 'case', 'config': 'config', 'reference': 'reference.vtk',
                       'completed_steps': 3, 'configuration': {'runtime_arguments': ['x=1'], 'expected_max_steps': None}}],
             'modes': [{'name': 'candidate', 'binary': binary.name}], 'threads': 4,
             'selected_gpu_uuid': 'GPU-fixture', 'run_timeout_seconds': timeout}
    monkeypatch.setattr(profiling, 'check', lambda path: state)
    validated = []
    def validate_log(log, limit, *, expected_completed_steps):
        assert 'ALL_STEPS_COMPLETE' in log
        assert expected_completed_steps == 3
        validated.append(log)
        return {'completed_steps': 3}
    monkeypatch.setattr(profiling, 'validate_log', validate_log)
    monkeypatch.setattr(profiling, 'read_vtk', lambda path: Path(path).read_text())
    monkeypatch.setattr(profiling, 'compare_vtk', lambda reference, actual: {'passed': actual == 'complete fields'})
    monkeypatch.setattr(profiling, 'parse_nsys_sqlite', lambda path: {'available': True})
    output = tmp_path / 'profile output'
    return root, output, profiler, validated


def test_waits_for_original_application_after_profiler_exits(tmp_path, monkeypatch):
    root, output, profiler, validated = fixture(tmp_path, monkeypatch)
    record = profiling.profile(root, output, case='case', mode='candidate', profiler=profiler, seconds=1)
    assert record['status'] == 'passed' and record['application_completed']
    assert record['profiler_exit_code'] == record['application_exit_code'] == 0
    assert record['completion_wait_seconds'] > 0
    assert record['profiled_execution_wall_seconds'] > record['profiler_wall_seconds']
    assert len(validated) == 1
    assert (output / 'profiler-returned-before-completion').read_text() == 'yes'
    assert 'ALL_STEPS_COMPLETE' in (output / 'application.log').read_text()
    assert 'ALL_STEPS_COMPLETE' not in (output / 'profiler.log').read_text()
    assert (output / 'execution-count').read_text() == 'once\n'
    assert read(output / 'application-completion.json')['exit_code'] == 0
    assert record['application_argv'] == [str(root / 'application'), 'x=1']
    assert '--duration=1' in record['argv'] and '--kill=none' in record['argv']
    assert not record['phase_timer_enabled'] and not record['timing_sample']


@pytest.mark.parametrize('profiler_mode', ['early', 'forward'])
def test_original_application_failure_preserves_exit_status_without_replay(tmp_path, monkeypatch, profiler_mode):
    root, output, profiler, validated = fixture(tmp_path, monkeypatch, exit_code=7, profiler_mode=profiler_mode)
    with pytest.raises(RuntimeError, match='application failed with exit code 7'):
        profiling.profile(root, output, case='case', mode='candidate', profiler=profiler, seconds=1)
    record = read(output / 'report.json')
    assert record['status'] == 'failed' and record['application_completed']
    assert record['application_exit_code'] == 7
    assert record['profiler_exit_code'] == (7 if profiler_mode == 'forward' else 0)
    assert (output / 'execution-count').read_text() == 'once\n'
    assert not validated


def test_profiler_failure_does_not_invent_application_completion(tmp_path, monkeypatch):
    root, output, profiler, validated = fixture(tmp_path, monkeypatch, profiler_mode='failure')
    with pytest.raises(RuntimeError, match='bounded profiler failed'):
        profiling.profile(root, output, case='case', mode='candidate', profiler=profiler, seconds=1)
    record = read(output / 'report.json')
    assert record['status'] == 'failed' and record['profiler_exit_code'] == 9
    assert not record['application_completed'] and not validated


def test_successful_profiler_without_target_completion_times_out(tmp_path, monkeypatch):
    root, output, profiler, validated = fixture(tmp_path, monkeypatch, profiler_mode='missing', timeout=0.1)
    with pytest.raises(TimeoutError, match='did not publish completion'):
        profiling.profile(root, output, case='case', mode='candidate', profiler=profiler, seconds=1)
    record = read(output / 'report.json')
    assert record['status'] == 'failed' and not record['application_completed']
    assert record['profiler_exit_code'] == 0 and not validated


def test_profiler_uses_same_total_timeout(tmp_path, monkeypatch):
    root, output, profiler, validated = fixture(tmp_path, monkeypatch, profiler_mode='timeout', timeout=0.1)
    with pytest.raises(profiling.subprocess.TimeoutExpired):
        profiling.profile(root, output, case='case', mode='candidate', profiler=profiler, seconds=1)
    assert read(output / 'report.json')['status'] == 'failed'
    assert not validated


@pytest.mark.parametrize('profiler_mode', ['non-object', 'malformed'])
def test_invalid_completion_has_clear_failure_receipt(tmp_path, monkeypatch, profiler_mode):
    root, output, profiler, validated = fixture(tmp_path, monkeypatch, profiler_mode=profiler_mode)
    with pytest.raises(RuntimeError, match='invalid profiled application completion marker'):
        profiling.profile(root, output, case='case', mode='candidate', profiler=profiler, seconds=1)
    record = read(output / 'report.json')
    assert record['status'] == 'failed' and not record['application_completed']
    assert record['failure_reason'] == 'invalid profiled application completion marker'
    assert not validated


@pytest.mark.parametrize('value', [
    {'schema_version': 1, 'token': 'foreign', 'exit_code': 0},
    {'schema_version': 1, 'token': 'expected', 'exit_code': True},
])
def test_completion_requires_this_target_and_an_integer_exit_status(tmp_path, value):
    marker = tmp_path / 'complete.json'
    write(marker, value)
    with pytest.raises(RuntimeError, match='invalid.*completion'):
        profiling.wait_completion(marker, 'expected', time.monotonic()+1)
