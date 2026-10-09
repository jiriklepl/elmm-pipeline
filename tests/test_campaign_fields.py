"""Full application dumps must retain precision, shape and original bounds."""

import numpy as np
import pytest

from elmm_pipeline.campaigns.halos import compare_fields, fields


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
