"""Opaque scalar CALL reservations retain source metadata and no payload credit."""
from copy import deepcopy
import json

import pytest

from compiler.tests.test_hierarchical_scalar_elements import SOURCE_ELEMENTS, emit
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from test_conditional_return_record import seal


def source(variant):
    value = SOURCE_ELEMENTS
    if variant == 'selector_write':
        value = value.replace('do i=1,n\na(i)=scale*b(i)', 'selector=1-selector\ndo i=1,n\na(i)=scale*b(i)')
    elif variant == 'two_calls':
        call = 'call consumer(left,right,n,ratios(selector))'
        value = value.replace(call, call + '\nselector=1-selector\n' + call)
    elif variant == 'negative_literal':
        value = value.replace('ratios(0:1)=[2.0_8,3.0_8]', 'ratios(-1:1)=[2.0_8,3.0_8,4.0_8]')
        value = value.replace('ratios(selector)', 'ratios(-1)')
    return value


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    result = {}
    for variant in ('ordinary', 'selector_write', 'two_calls', 'negative_literal'):
        directory = tmp_path_factory.mktemp('scalar-element-' + variant)
        _path, builder, _middle, _calls, _outputs, report = emit(directory, source(variant))
        assert builder.finish()
        result[variant] = json.loads(json.dumps(report))
        (directory / 'report.json').write_text(json.dumps(result[variant], indent=2) + '\n')
    return result


def parts(report):
    scope = report['scopes'][0]
    plan = scope['checked_execution_plan']
    resource = next(row for row in scope['resource_bindings']['resources'] if row['resource'] == 'hierarchical::ratios')
    consumer = next(p for p in plan['procedures'] if p['procedure'] == 'hierarchical::consumer')
    site = consumer['calls'][0]
    return scope, plan, resource, site, site['original_scalar_element_coordinates']


def seal_sites(report, *, proof=False):
    scope, _plan, _resource, site, coordinates = parts(report)
    if proof:
        coordinates['proof_identity'] = _digest({k: v for k, v in coordinates.items() if k != 'proof_identity'})
        site['coordinate_proof_identity'] = coordinates['proof_identity']
    coordinator = next(row for row in scope['module_coordinators'] if row['procedure'] == 'hierarchical::consumer')
    coordinator['calls'][0] = deepcopy(site)
    seal(report)


@pytest.mark.parametrize('variant', ['ordinary', 'selector_write', 'two_calls', 'negative_literal'])
def test_authentic_scalar_element_reservation_is_readonly_transport(originals, variant):
    report = deepcopy(originals[variant])
    before = deepcopy(report)
    validate_checked_execution_plans(report)
    assert report == before
    scope, plan, resource, _site, _proof = parts(report)
    canonical = next(row for row in plan['resources'] if row['resource'] == resource['resource'])
    assert {key: resource[key] for key in canonical} == canonical
    assert resource['managed_payload'] is resource['definition_authority'] is False
    assert scope['checked_execution_plan']['storage_epoch']['return_contract']['version'] == 2


@pytest.mark.parametrize('fault', [
    'role', 'managed', 'defined', 'unknown_row', 'missing_role', 'regid', 'descriptor_rank', 'descriptor_type',
    'descriptor_bounds', 'descriptor_foreign', 'unknown_descriptor', 'unused_reservation', 'managed_epoch',
    'unit_backing', 'missing_coordinates', 'wrong_coordinate_identity', 'foreign_caller', 'foreign_graph',
    'wrong_node', 'wrong_guard_digest', 'wrong_source', 'coordinate_execution', 'coordinate_bounds',
    'unknown_coordinate', 'selector', 'initializer', 'element_bounds', 'element_definition', 'element_payload',
    'unknown_element', 'call_actual', 'call_target', 'closure_version', 'closure_backing', 'closure_readonly', 'closure_effect',
    'closure_formal', 'closure_source', 'missing_resolved_call', 'wrong_call_slot',
])
def test_resealed_opaque_inventory_cannot_invent_or_manage_scalar_elements(originals, fault):
    report = deepcopy(originals['ordinary'])
    validate_checked_execution_plans(report)
    scope, plan, resource, site, proof = parts(report)
    element = proof['original_scalar_elements'][0]
    epoch = plan['storage_epoch']
    resign_proof = False
    if fault == 'role': resource['role'] = 'managed_array'
    elif fault == 'managed': resource['managed_payload'] = True
    elif fault == 'defined': resource['definition_authority'] = True
    elif fault == 'unknown_row': resource['initializer_authority'] = True
    elif fault == 'missing_role':
        for key in ('role', 'managed_payload', 'definition_authority'): resource.pop(key)
    elif fault == 'regid': resource['registration_identity'] += 1
    elif fault.startswith('descriptor_') or fault == 'unknown_descriptor':
        descriptor = resource['descriptor']
        key, value = {'descriptor_rank':('rank', 2), 'descriptor_type':('type', 'integer'),
                      'descriptor_bounds':('lower_bounds', ['-1']), 'descriptor_foreign':('resource', 'other::ratios'),
                      'unknown_descriptor':('payload_authority', True)}[fault]
        descriptor[key] = value
    elif fault == 'unused_reservation':
        extra = deepcopy(resource)
        extra['resource'] = extra['descriptor']['resource'] = 'hierarchical::unused'
        extra['descriptor']['name'] = 'unused'
        extra['registration_identity'] = 99
        scope['resource_bindings']['resources'].append(extra)
        base = {k: extra[k] for k in ('resource', 'descriptor', 'registration_identity')}
        plan['resources'].append(deepcopy(base))
        epoch['resources'].append({**base, 'device_replica_authorized':False, 'initialization_authorized':False,
                                   'storage':'opaque host reservation', 'lifetime':'original checked native access'})
        epoch['resources'].sort(key=lambda row: row['resource'])
    elif fault == 'managed_epoch':
        row = next(r for r in epoch['resources'] if r['resource'] == resource['resource'])
        row.pop('initialization_authorized')
        row.update(device_replica_authorized=True, storage='stable', lifetime='selected_window',
                   allocation_changes=False, escapes=False)
    elif fault == 'unit_backing': plan['procedures'][1]['units'][0]['resources']['forged'] = resource['resource']
    elif fault == 'missing_coordinates': site.pop('original_scalar_element_coordinates')
    elif fault == 'wrong_coordinate_identity': site['coordinate_proof_identity'] = 'f'*64
    elif fault in {'foreign_caller', 'foreign_graph', 'wrong_node', 'wrong_source', 'coordinate_execution',
                   'coordinate_bounds', 'unknown_coordinate', 'wrong_guard_digest'}:
        key, value = {'foreign_caller':('caller', 'hierarchical::producer'), 'foreign_graph':('structured_identity', 'f'*64),
                      'wrong_node':('node_id', 'hierarchical::middle/body/2'), 'wrong_source':('source_identity', {}),
                      'coordinate_execution':('execution_authorized', True), 'coordinate_bounds':('bounds_evaluated', True),
                      'unknown_coordinate':('entry_fact_authority', True), 'wrong_guard_digest':('guard', ['IF (.FALSE.)'])}[fault]
        proof[key] = value
        resign_proof = fault != 'wrong_guard_digest'
    elif fault in {'selector', 'initializer', 'element_bounds', 'element_definition', 'element_payload', 'unknown_element'}:
        if fault == 'selector': element['index']['expression'] = 'different_selector'
        elif fault == 'initializer': element['declaration_initializer'] = '= [9.0_8, 10.0_8]'
        elif fault == 'element_bounds': element['bounds'][0] = -1
        elif fault == 'element_definition': element['definition_authority'] = True
        elif fault == 'element_payload': element['managed_storage_authority'] = True
        else: element['execution_authority'] = True
        resign_proof = True
    elif fault in {'call_actual', 'call_target'}:
        if fault == 'call_actual': proof['call']['actual_arguments'][-1] = 'ratios(0)'
        else: proof['call']['target'] = 'other_consumer'
        resign_proof = True
    elif fault.startswith('closure_'):
        closure = proof['scalar_element_closure']
        if fault == 'closure_version': closure['version'] = 1
        elif fault == 'closure_backing': closure['backing_resources'] = []
        elif fault == 'closure_readonly': closure['only_scalar_in_forwarding'] = False
        elif fault == 'closure_effect': closure['effect_authority'] = True
        elif fault == 'closure_formal': closure['procedures'][0]['associated_formals'] = []
        else: closure['procedures'][0]['source_sha256'] = 'f'*64
        resign_proof = True
    elif fault == 'missing_resolved_call':
        report['resolved_calls'] = [row for row in report['resolved_calls'] if row != proof['call']]
    else:
        slot = next(iter(site['handle_bindings']))
        site['handle_bindings'][slot] += 1
    # Absence itself is a negative, so avoid attempting to reseal a missing proof.
    if fault == 'missing_coordinates':
        coordinator = next(row for row in scope['module_coordinators'] if row['procedure'] == 'hierarchical::consumer')
        coordinator['calls'][0] = deepcopy(site)
        seal(report)
    else:
        seal_sites(report, proof=resign_proof)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)


def test_same_callee_second_call_cannot_borrow_first_original_coordinate_use(originals):
    report = deepcopy(originals['two_calls'])
    validate_checked_execution_plans(report)
    scope, plan, _resource, first, proof = parts(report)
    consumer = next(p for p in plan['procedures'] if p['procedure'] == 'hierarchical::consumer')
    assert len(consumer['calls']) == 2
    second = consumer['calls'][1]
    second['original_scalar_element_coordinates'] = deepcopy(proof)
    second['coordinate_proof_identity'] = first['coordinate_proof_identity']
    coordinator = next(row for row in scope['module_coordinators'] if row['procedure'] == 'hierarchical::consumer')
    coordinator['calls'][1] = deepcopy(second)
    seal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)
