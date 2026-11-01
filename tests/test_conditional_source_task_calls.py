"""A conditional owner preserves distinct source-CALL local namespaces."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_same_family_native_call import source_fixture
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans


HELPERS = '''subroutine helper(a,p,n,flag)
real(8),intent(inout)::a(:)
real(8),intent(in)::p(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
!$omp parallel private(i)
if(flag.and.n>=6)then
!$omp sections
!$omp section
do i=n-2,n
a(i)=p(i-n+3)
enddo
!$omp section
do i=1,3
a(i)=p(i+n-3)
enddo
!$omp end sections nowait
endif
!$omp end parallel
end subroutine
subroutine relay(a,p,n,flag)
real(8),intent(inout)::a(:)
real(8),intent(in)::p(:)
integer,intent(in)::n
logical,intent(in)::flag
call helper(a,p,n,flag)
end subroutine
'''


def combined_source():
    return source_fixture().replace(
        'call native_worker(left,right,n,scale)',
        'call native_worker(left,right,n,scale)\n'
        'call relay(left,right,n,change)\ncall relay(right,left,n,change)',
    ).replace('end module', HELPERS + 'end module')


@pytest.fixture(scope='module')
def authentic(tmp_path_factory):
    path, builder = fixture(tmp_path_factory.mktemp('conditional-source-calls'), combined_source())
    outputs, report = builder.run()
    assert report['scope_count'] == 1
    manifest = json.loads(outputs['scope-manifest.json'])
    validate_checked_execution_plans(manifest)
    return manifest


def target(manifest):
    plan = manifest['scopes'][0]['checked_execution_plan']
    returned = plan['storage_epoch']['return_contract']
    route = next(route for member in returned['members'] if member['procedure'] == 'hierarchical::relay'
                 for route in member['routes'] if route['kind'] == 'checked_child')
    task = next(task for member in plan['storage_epoch']['members']
                for task in member.get('source_device_tasks', [])
                if task['kind'] == 'original_atomic_conditional_joined_team')
    return returned, route, task


def test_same_family_retirement_and_projected_section_round_trip(authentic):
    before = deepcopy(authentic)
    validate_checked_execution_plans(authentic)
    returned, route, task = target(authentic)
    assert returned['version'] == 3 and task['version'] == 4
    assert any('call_preservation' in value for member in returned['members'] for value in member['routes'])
    edge = next(e for e in task['execution_binding']['call_projection']['calls'] if e['caller'] == 'hierarchical::relay')
    assert edge['source_call']['resource_mapping']['argument::a'] == 'argument::a'
    assert edge['resource_mappings']['argument::a'] == 'hierarchical::left'
    assert route['handle_forwarding'] != route['canonical_handles']
    assert authentic == before


@pytest.mark.parametrize('fault', ['wrong_parent_slot', 'changed_source_call', 'conflicting_edge', 'changed_selection'])
def test_conditional_projection_keeps_exact_call_and_slot_rejection(authentic, fault):
    bad = deepcopy(authentic)
    returned, route, task = target(bad)
    projection = task['execution_binding']['call_projection']
    edge = next(e for e in projection['calls'] if e['caller'] == 'hierarchical::relay')
    if fault == 'wrong_parent_slot':
        from elmm_pipeline.conditional_return_record import _slot
        key = next(iter(route['handle_forwarding']))
        route['handle_forwarding'][key] = _slot('hierarchical::relay', 'hierarchical::left')
    elif fault == 'changed_source_call':
        edge['source_call']['resource_mapping']['argument::a'] = 'argument::p'
    elif fault == 'conflicting_edge':
        projection['calls'].append(deepcopy(edge))
    else:
        edge['selected_node_ids'] = ['hierarchical::relay/body/99']
    projection['identity'] = _digest({key: value for key, value in projection.items() if key != 'identity'})
    task['identity'] = _digest({key: value for key, value in task.items() if key != 'identity'})
    returned['identity'] = _digest({key: value for key, value in returned.items() if key != 'identity'})
    with pytest.raises(ValueError):
        validate_checked_execution_plans(bad)
