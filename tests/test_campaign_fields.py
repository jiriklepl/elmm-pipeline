"""Full application dumps must retain precision, shape and original bounds."""

from types import SimpleNamespace

import numpy as np
import pytest

from elmm_pipeline.campaigns import halos
from elmm_pipeline.campaigns.halos import compare_fields, fields
from elmm_pipeline.campaigns.identity import read

DRIVER = {'dtype': 'float64', 'field_count': 1, 'rtol': 1e-5, 'atol': 1e-6,
          'metadata_format': 'storage-shape-bounds-v1'}


def dump(directory, metadata='64 2 3 -2 7 -1 9', values=None):
    directory.mkdir(exist_ok=True)
    path = directory / 'fields-state.bin'
    np.asarray(np.arange(6) if values is None else values, dtype=np.float64).tofile(path)
    path.with_suffix('.shape').write_text(metadata)
    return directory


def test_complete_fields_preserve_negative_bounds_and_exact_coverage(tmp_path):
    reference = fields(dump(tmp_path/'native'), DRIVER)
    candidate = fields(dump(tmp_path/'candidate'), DRIVER)
    row = compare_fields(candidate, reference, DRIVER)['fields-state.bin']
    assert row['passed'] and row['values'] == 6
    assert row['layout']['lower_bounds'] == [-2, 7]
    # A halo value beyond tolerance must fail even when all metadata matches.
    candidate['fields-state.bin'][0][-1] += 1
    assert not compare_fields(candidate, reference, DRIVER)['fields-state.bin']['passed']


@pytest.mark.parametrize('metadata', [
    '32 2 3 -2 7 -1 9',  # same bytes cannot assert another element precision
    '64 2 3 -2 7 -1',  # truncated descriptor
    '64 -2 3 -2 7 -5 9',  # negative extent
    '64 2 3 -2 7 0 9',  # inconsistent upper bound
    '64 2147483648 2147483648 1 1 2147483648 2147483648',  # no product overflow
])
def test_complete_field_metadata_cannot_hide_shape_or_precision_mismatch(tmp_path, metadata):
    with pytest.raises(ValueError, match='complete field'):
        fields(dump(tmp_path, metadata), DRIVER)


def test_same_shape_different_logical_bounds_is_a_mismatch(tmp_path):
    reference = fields(dump(tmp_path/'native'), DRIVER)
    candidate = fields(dump(tmp_path/'candidate', '64 2 3 1 1 2 3'), DRIVER)
    with pytest.raises(ValueError, match='original bounds'):
        compare_fields(candidate, reference, DRIVER)


def test_empty_fields_still_require_their_descriptors(tmp_path):
    directory = dump(tmp_path, '64 0 3 1 7 0 9', values=[])
    reference = fields(directory, DRIVER)
    assert compare_fields(reference, reference, DRIVER)['fields-state.bin']['values'] == 0
    (directory/'fields-state.shape').unlink()
    with pytest.raises(FileNotFoundError):
        fields(directory, DRIVER)


def test_trailing_partial_element_and_missing_field_are_rejected(tmp_path):
    directory = dump(tmp_path)
    with (directory/'fields-state.bin').open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='partial element'):
        fields(directory, DRIVER)
    (directory/'fields-state.bin').unlink()
    with pytest.raises(ValueError, match='missing complete fields'):
        fields(directory, DRIVER)


def test_fixed_layout_fixture_format_remains_supported(tmp_path):
    directory = dump(tmp_path)
    (directory/'fields-state.shape').unlink()
    driver = {key: value for key, value in DRIVER.items() if key != 'metadata_format'}
    reference = fields(directory, driver)
    assert compare_fields(reference, reference, driver)['fields-state.bin']['passed']


@pytest.mark.parametrize('name', ['atol', 'rtol'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -1.])
def test_complete_fields_reject_invalid_tolerance_even_for_identical_values(name, value):
    observed = {'state': (np.asarray([1.]), None)}
    with pytest.raises(ValueError, match='tolerances must be finite and nonnegative'):
        compare_fields(observed, observed, {**DRIVER, name: value})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('side', ['candidate', 'reference', 'both'])
def test_complete_fields_reject_and_record_nonfinite_values(value, side):
    import json

    actual, expected = np.arange(6, dtype=np.float64), np.arange(6, dtype=np.float64)
    if side in ('candidate', 'both'):
        actual[2] = value
    if side in ('reference', 'both'):
        expected[2] = value
    row = compare_fields({'state': (actual, None)}, {'state': (expected, None)}, DRIVER)['state']
    assert not row['passed'] and row['failed_values'] == 1
    assert row['candidate_nonfinite_values'] == int(side in ('candidate', 'both'))
    assert row['reference_nonfinite_values'] == int(side in ('reference', 'both'))
    assert row['absolute_error_nonfinite_values'] == 1 and row['max_absolute_error'] is None
    assert row['failed_flat_indices'] == [2] and not row['failed_locations_truncated']
    assert (row['atol'], row['rtol']) == (DRIVER['atol'], DRIVER['rtol'])
    json.dumps(row, allow_nan=False)


def test_complete_field_failure_locations_are_bounded_and_tolerance_is_unchanged():
    actual, expected = np.ones(20), np.zeros(20)
    row = compare_fields({'state': (actual, None)}, {'state': (expected, None)}, DRIVER)['state']
    assert not row['passed'] and row['failed_values'] == 20
    assert row['failed_flat_indices'] == list(range(8)) and row['failed_locations_truncated']
    assert row['max_absolute_error'] == 1
    assert row['candidate_nonfinite_values'] == row['reference_nonfinite_values'] == 0
    assert compare_fields({'state': (np.asarray([0.5e-6]), None)},
                          {'state': (np.asarray([0.]), None)}, DRIVER)['state']['passed']


def test_finite_subtraction_overflow_has_serializable_failure_evidence():
    import json

    maximum = np.finfo(np.float64).max
    row = compare_fields({'state': (np.asarray([maximum]), None)},
                         {'state': (np.asarray([-maximum]), None)}, DRIVER)['state']
    assert not row['passed'] and row['failed_values'] == 1
    assert row['candidate_nonfinite_values'] == row['reference_nonfinite_values'] == 0
    assert row['absolute_error_nonfinite_values'] == 1 and row['max_absolute_error'] is None
    json.dumps(row, allow_nan=False)


def mock_validation(monkeypatch, values):
    """Exercise the complete suite controller without compiling application code."""
    drivers = []
    for name, modes in values.items():
        drivers.append({**DRIVER, 'name': name, 'coverage': 'complete fields and halos',
                        'sources': {mode: mode + '.f90' for mode in modes},
                        'link': {mode: {'include_dirs': [], 'objects': [], 'flags': []} for mode in modes},
                        'flags': [], 'success_marker': 'OK'})
    state = {'threads': 4, 'selected_gpu_uuid': 'GPU-fixture', 'halo_drivers': drivers,
             'tools': {'fortran': {'path': '/fixture/fortran'}}}
    monkeypatch.setattr(halos, 'check', lambda root: state)
    monkeypatch.setattr(halos.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout='OK', stderr=''))
    seen = []

    def observed(directory, driver):
        mode = directory.name
        seen.append((driver['name'], mode))
        return {'fields-state.bin': (np.asarray([values[driver['name']][mode]], dtype=np.float64), None)}

    monkeypatch.setattr(halos, 'fields', observed)
    return seen


def test_suite_retains_all_mismatches_and_keeps_original_native_reference(tmp_path, monkeypatch):
    seen = mock_validation(monkeypatch, {
        'first': {'bad': 2., 'native': 0., 'good': 0., 'also-bad': 2.},
        'second': {'bad': 3., 'native': 1., 'good': 1.}})
    with pytest.raises(RuntimeError, match='all comparisons retained'):
        halos.run(tmp_path)
    report = read(tmp_path / 'halos/report.json')
    assert report['status'] == 'failed' and report['performance_measured'] is False
    assert seen == [('first', 'native'), ('first', 'bad'), ('first', 'good'), ('first', 'also-bad'),
                    ('second', 'native'), ('second', 'bad'), ('second', 'good')]
    assert [(row['driver'], row['mode']) for row in report['failures']] == [
        ('first', 'bad'), ('first', 'also-bad'), ('second', 'bad')]
    assert all(row['fields'] == ['fields-state.bin'] for row in report['failures'])
    assert [len(row['runs']) for row in report['drivers']] == [4, 3]
    assert report['drivers'][0]['runs'][2]['comparisons']['fields-state.bin']['passed']


def test_suite_without_mismatches_passes(tmp_path, monkeypatch):
    mock_validation(monkeypatch, {'fixture': {'candidate': 1., 'native': 1.}})
    halos.run(tmp_path)
    report = read(tmp_path / 'halos/report.json')
    assert report['status'] == 'passed' and report['failures'] == []


def test_matching_nonfinite_reference_fails_suite_and_blocks_timing(tmp_path, monkeypatch):
    import importlib
    import os

    from elmm_pipeline.campaigns.identity import digest, write

    mock_validation(monkeypatch, {'fixture': {'native': float('inf'), 'candidate': float('inf')}})
    with pytest.raises(RuntimeError, match='all comparisons retained'):
        halos.run(tmp_path)
    report = read(tmp_path / 'halos/report.json')
    assert report['status'] == 'failed'
    assert [row['mode'] for row in report['failures']] == ['native', 'candidate']
    assert all(not run['comparisons']['fields-state.bin']['passed']
               for run in report['drivers'][0]['runs'])
    runner = importlib.import_module('elmm_pipeline.campaigns.run')
    write(tmp_path / 'campaign.json', {})
    write(tmp_path / 'verification/report.json', {
        'status': 'passed', 'campaign_sha256': digest(tmp_path / 'campaign.json')})
    monkeypatch.setattr(runner, 'check', lambda root: {'host': {'affinity': sorted(os.sched_getaffinity(0))}})
    monkeypatch.setattr(runner, 'gpu_identity', lambda *args: None)
    monkeypatch.setattr(runner, 'run_one', lambda *args, **kwargs: pytest.fail('timing must not start'))
    with pytest.raises(ValueError, match='halo verification must pass before timing'):
        runner.run(tmp_path, 'timings')


@pytest.mark.parametrize('failure', ['build', 'execution', 'metadata'])
def test_suite_stops_on_noncomparison_errors_and_records_failure(tmp_path, monkeypatch, failure):
    seen = mock_validation(monkeypatch, {'fixture': {'native': 0., 'candidate': 0.}})
    if failure == 'metadata':
        def malformed(*args):
            raise ValueError('complete field extent or byte count mismatch')
        monkeypatch.setattr(halos, 'fields', malformed)
    else:
        calls = []

        def fail(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=int(failure == 'build' or len(calls) == 2),
                                   stdout='OK', stderr='failure fixture')

        monkeypatch.setattr(halos.subprocess, 'run', fail)
    with pytest.raises((RuntimeError, ValueError)):
        halos.run(tmp_path)
    report = read(tmp_path / 'halos/report.json')
    assert report['status'] == 'failed' and report['error']['message']
    assert report['drivers'][0]['runs'] == [] and not seen
    assert not (tmp_path / 'halos/fixture/candidate').exists()


def test_failed_halos_cannot_start_timing(tmp_path, monkeypatch):
    import importlib
    import os

    from elmm_pipeline.campaigns.identity import digest, write

    runner = importlib.import_module('elmm_pipeline.campaigns.run')
    write(tmp_path / 'campaign.json', {})
    write(tmp_path / 'verification/report.json', {
        'status': 'passed', 'campaign_sha256': digest(tmp_path / 'campaign.json')})
    write(tmp_path / 'halos/report.json', {'status': 'failed'})
    monkeypatch.setattr(runner, 'check', lambda root: {'host': {'affinity': sorted(os.sched_getaffinity(0))}})
    monkeypatch.setattr(runner, 'gpu_identity', lambda *args: None)
    monkeypatch.setattr(runner, 'run_one', lambda *args, **kwargs: pytest.fail('timing must not start'))
    with pytest.raises(ValueError, match='halo verification must pass before timing'):
        runner.run(tmp_path, 'timing')
