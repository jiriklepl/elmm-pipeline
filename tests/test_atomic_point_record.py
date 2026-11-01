"""Predicate-only host reads retain original parent/graph linkage."""
from copy import deepcopy

import pytest
from compiler.tests.test_atomic_point_predicates import point_source
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from test_atomic_source_task_record import reseal, task

from compiler.tests.test_atomic_joined_team_source import emit
from compiler.tests.test_source_scopes import FACT


@pytest.fixture
def issued(tmp_path):
    _outputs, report, _text = emit(tmp_path, source=point_source(),
                                  extra_captures={'argument::gate': dict(FACT)})
    return report


def test_authentic_fixed_integer_predicate_record_round_trips(issued):
    validate_checked_execution_plans(issued)
    record = task(issued)
    assert record['version'] == record['source_proof']['version'] == 2
    assert all('argument::gate' not in [field['resource'] for field in item['provider']['metadata']['fields']]
               for item in record['alternatives'])


@pytest.mark.parametrize('fault', ['point', 'lower', 'type', 'unwritten', 'guard', 'missing', 'nonlist',
                                 'foreign', 'parameter', 'unknown', 'bool_index', 'version', 'gpu_field'])
def test_resealed_predicate_metadata_cannot_change_parent_points_or_gpu_fields(issued, fault):
    record = task(issued)
    team = record['source_proof']
    point = team['predicates'][0]['host_point_reads'][0]
    parent = team['parent_completion']['uniform_array_reads'][0]
    if fault == 'point':
        point['subscript'] += 1
    elif fault == 'lower':
        parent['original_lower_bound'] = 20
    elif fault == 'type':
        parent['type'] = 'real'
    elif fault == 'unwritten':
        parent['shared_and_unwritten_in_complete_team'] = False
    elif fault == 'guard':
        team['parent_completion']['guarded_assumed_shape_array_conditions_authorized'] = True
    elif fault == 'missing':
        team['predicates'][0]['host_point_reads'] = []
    elif fault == 'nonlist':
        team['predicates'][0]['host_point_reads'] = True
    elif fault == 'foreign':
        point['resource'] = 'other::gate'
    elif fault == 'parameter':
        point['parameter_resource'] = 'other::index'
    elif fault == 'unknown':
        point['authority'] = True
    elif fault == 'bool_index':
        point['subscript'] = parent['subscript'] = True
    elif fault == 'version':
        record['version'] = 1
    elif fault == 'gpu_field':
        descriptor = next(item for item in team['bindings'] if item['resource'] == 'argument::gate')
        record['alternatives'][0]['provider']['metadata']['fields'].append(deepcopy(descriptor))
    reseal(issued)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(issued)


@pytest.mark.parametrize('shape', ['nested', 'siblings'])
@pytest.mark.parametrize('fault', ['move', 'move_both', 'swap_headers', 'omit_site', 'duplicate_site', 'reference'])
def test_exact_original_header_sites_prevent_point_migration(tmp_path, shape, fault):
    from compiler.tests.test_atomic_worksharing_team import BODY, unit

    source = point_source()
    if shape == 'siblings':
        source = source.replace(BODY.replace('mode==4', 'gate(index)==4').replace('if(flag)then', 'if(gate(2)>0)then'),
            'if(gate(index)==4)then\n' + unit('a', 'a(i)+p(i)') + 'endif\n'
            + 'if(gate(2)>0)then\n' + unit('b', 'b(i)+p(i)') + 'endif\n')
    _outputs, report, _text = emit(tmp_path, source=source, extra_captures={'argument::gate': dict(FACT)})
    validate_checked_execution_plans(report)
    team = task(report)['source_proof']
    predicates, sites = team['predicates'], team['predicate_read_sites']
    if fault in ('move', 'move_both'):
        predicates[0]['host_point_reads'], predicates[1]['host_point_reads'] = predicates[1]['host_point_reads'], predicates[0]['host_point_reads']
        if fault == 'move_both':
            sites[0]['host_point_reads'], sites[1]['host_point_reads'] = sites[1]['host_point_reads'], sites[0]['host_point_reads']
    elif fault == 'swap_headers':
        predicates[0], predicates[1] = predicates[1], predicates[0]
        sites[0], sites[1] = sites[1], sites[0]
    elif fault == 'omit_site':
        sites.pop()
    elif fault == 'duplicate_site':
        sites[1] = deepcopy(sites[0])
    elif fault == 'reference':
        predicates[0]['host_point_reads'][0]['source_reference'] = 'other_gate(index)'
        sites[0]['host_point_reads'][0]['source_reference'] = 'other_gate(index)'
    reseal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)
