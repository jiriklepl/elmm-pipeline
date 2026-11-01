"""Authentic SECTION metadata keeps one complete typed pre-admission guard."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_conditional_sections import ENTRY, SOURCE
from compiler.tests.test_source_scopes import FACT


def task(report):
    return report['scopes'][0]['source_device_tasks'][0]


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp('conditional-sections-reader')/'source.f90'
    path.write_text(SOURCE)
    builder = ScopeBuilder([path], ENTRY, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', scope_execution='graph', host_threads=4),
        facts={'schema_version': 1, 'participation': 'serial',
               'sources': {str(path): sha256(path.read_bytes()).hexdigest()},
               'captures': {'argument::a': dict(FACT)}})
    outputs, _report = builder.run()
    report = json.loads(outputs['scope-manifest.json'])
    validate_checked_execution_plans(report)
    (path.parent/'scope-manifest.json').write_text(outputs['scope-manifest.json'])
    return report


def reseal(report):
    record = task(report)
    team = record['source_proof']
    team['identity'] = _digest({k: v for k, v in team.items() if k != 'identity'})
    for variant in record['alternatives']:
        source, prep, effect, read = [variant[k] for k in ('source', 'preparation', 'physical_effects', 'source_reads')]
        context = prep.get('conditional_numerical_context')
        if isinstance(context, dict):
            context['team_identity'] = team['identity']
            for path in context.get('paths', []):
                domain = path.get('domain')
                if isinstance(domain, dict):
                    for clause in domain.get('guard_conjunction', []):
                        if isinstance(clause, dict):
                            clause['identity'] = _digest({k: v for k, v in clause.items() if k != 'identity'})
            context['identity'] = _digest({k: v for k, v in context.items() if k != 'identity'})
            for row in (effect, read, variant['provider']['metadata']):
                row['conditional_numerical_context_identity'] = context['identity']
        source['team_identity'] = team['identity']
        source['proof_identity'] = _digest({k: v for k, v in source.items() if k != 'proof_identity'})
        prep['completion'] = deepcopy(source)
        effect['proof_identity'] = _digest({k: v for k, v in effect.items() if k != 'proof_identity'})
        read['numerical_effect_identity'] = effect['proof_identity']
        read['proof_identity'] = _digest({k: v for k, v in read.items() if k != 'proof_identity'})
        metadata = variant['provider']['metadata']
        metadata.update(team_identity=team['identity'], alternative_identity=source['proof_identity'],
                        physical_effect_identity=effect['proof_identity'], source_read_identity=read['proof_identity'])
    record['identity'] = _digest({k: v for k, v in record.items() if k != 'identity'})
    plan = report['scopes'][0]['checked_execution_plan']
    epoch = plan['storage_epoch']
    member = next(m for m in epoch['members'] if m['procedure'] == team['procedure'])
    member['source_device_tasks'] = [deepcopy(record)]
    epoch['identity'] = _digest({k: v for k, v in epoch.items() if k != 'identity'})
    plan['identity'] = _digest({k: v for k, v in plan.items() if k != 'identity'})


def test_real_section_graph_has_guard_linkage_without_independent_completion(baseline):
    record = task(baseline)
    assert record['version'] == 3
    assert record['source_proof']['alternatives'] == [[0, 1], []]
    variant, = record['alternatives']
    context = variant['preparation']['conditional_numerical_context']
    assert context['paths'][1]['domain'] is None
    assert context['paths'][0]['domain']['version'] == 2
    assert variant['provider']['metadata']['conditional_guard_before_source_read_admission'] is True
    assert record['source_proof']['parent_completion']['section_independence_established'] is False


_CONTEXT_FAULTS = [
    ('version', True), ('version', 2), ('kind', 'foreign'), ('source_graph', 'f'*64),
    ('selected_node_ids', []), ('source_inputs', {}), ('source_identity', 'f'*64),
    ('provider_identity', 'f'*64), ('extraction_identity', ''), ('payload_identity', ''),
    ('model_units', True), ('model_units', 1), ('source_work_budget', 65537),
    ('isl_operation_budget', 1000001), ('guard', 'after submission'), ('incoming_reads', 'all reads'),
    ('original_integer_prework_required', False), ('physical_bounds_and_alias_guards_required', False),
    ('independent_section_completion_authorized', True), ('definition_transfer_authorized', True),
    ('execution_authorized', True), ('source_read_admission_authorized', True), ('unknown', 1),
]


@pytest.mark.parametrize('key,value', _CONTEXT_FAULTS)
def test_resealed_context_cannot_change_source_or_authority(baseline, key, value):
    report = deepcopy(baseline)
    task(report)['alternatives'][0]['preparation']['conditional_numerical_context'][key] = value
    reseal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)


@pytest.mark.parametrize('fault', [
    'task_version', 'team_version', 'context_missing', 'context_required', 'unit_kind', 'unit_duplicate',
    'unit_marker_foreign', 'unit_marker_alias', 'unit_group_alias', 'parent_native', 'parent_count',
    'parent_independent', 'parent_internal_cut', 'path_missing', 'path_reordered', 'empty_authority',
    'domain_version', 'domain_authority', 'domain_units', 'domain_parameter', 'domain_digest',
    'domain_unknown', 'clause_duplicate', 'clause_unknown', 'clause_text', 'coefficient_bool',
    'coefficient_overflow', 'coefficient_foreign', 'coefficient_duplicate', 'constant_bool',
    'constant_overflow', 'equality_nonbool', 'parameter_axis', 'parameter_runtime', 'parameter_rank',
    'parameter_kind', 'parameter_id_duplicate', 'parameter_name_duplicate', 'provider_version',
    'provider_admission', 'provider_generation', 'provider_integer', 'provider_guard', 'provider_zero',
    'provider_parameter_order', 'read_guard', 'read_context', 'effect_context',
    'model_symbol_bool', 'path_symbol_bool', 'unit_missing', 'resource_nontext',
])
def test_resealed_section_transport_cannot_omit_or_rebind_guard(baseline, fault):
    report = deepcopy(baseline)
    record = task(report); team = record['source_proof']; variant = record['alternatives'][0]
    prep = variant['preparation']; context = prep['conditional_numerical_context']
    domain = context['paths'][0]['domain']; clause = domain['guard_conjunction'][0]
    condition = clause['constraints'][0][0]; parameter = context['parameters'][1]
    metadata = variant['provider']['metadata']
    if fault == 'task_version': record['version'] = 2
    elif fault == 'team_version': team['version'] = 2
    elif fault == 'context_missing': prep.pop('conditional_numerical_context')
    elif fault == 'context_required': team['conditional_affine_context_required'] = False
    elif fault == 'unit_kind': team['units'][0]['unit_kind'] = 'worksharing_do'
    elif fault == 'unit_duplicate': team['units'][1] = deepcopy(team['units'][0])
    elif fault == 'unit_marker_foreign': team['units'][0]['section_directive_node_id'] = 'foreign/body/1'
    elif fault == 'unit_marker_alias': team['units'][0]['section_directive_node_id'] = team['units'][0]['source_node_id']
    elif fault == 'unit_group_alias': team['units'][0]['section_group_node_id'] = team['units'][0]['source_node_id']
    elif fault == 'parent_native': team['parent_completion']['native_only'] = False
    elif fault == 'parent_count': team['parent_completion']['original_section_count'] = True
    elif fault == 'parent_independent': team['parent_completion']['section_independence_established'] = True
    elif fault == 'parent_internal_cut': team['parent_completion']['internal_cuts_authorized'] = True
    elif fault == 'path_missing': context['paths'].pop()
    elif fault == 'path_reordered': context['paths'].reverse()
    elif fault == 'empty_authority': context['paths'][1]['domain'] = deepcopy(domain)
    elif fault == 'domain_version': domain['version'] = 1
    elif fault == 'domain_authority': domain['source_authority'] = True
    elif fault == 'domain_units': domain['units'] = True
    elif fault == 'domain_parameter': domain['parameters'][0]['symbol_id'] = 8
    elif fault == 'domain_digest': domain['model_identity'] = ''
    elif fault == 'domain_unknown': domain['forbidden_relations'] = {}
    elif fault == 'clause_duplicate': domain['guard_conjunction'].append(deepcopy(clause))
    elif fault == 'clause_unknown': clause['authority'] = True
    elif fault == 'clause_text': clause['legal_set'] = ''
    elif fault == 'coefficient_bool': condition['coefficients'][0][1] = True
    elif fault == 'coefficient_overflow': condition['coefficients'][0][1] = 2**63
    elif fault == 'coefficient_foreign': condition['coefficients'][0][0] = 'p99'
    elif fault == 'coefficient_duplicate': condition['coefficients'].append(deepcopy(condition['coefficients'][0]))
    elif fault == 'constant_bool': condition['constant'] = True
    elif fault == 'constant_overflow': condition['constant'] = -(2**63)-1
    elif fault == 'equality_nonbool': condition['equality'] = 1
    elif fault == 'parameter_axis': parameter['lower_bound_dimension'] = True
    elif fault == 'parameter_runtime': parameter['runtime_lower_bound'] = False
    elif fault == 'parameter_rank': parameter['rank'] = 1
    elif fault == 'parameter_kind': parameter['dtype'] = 'real'
    elif fault == 'parameter_id_duplicate': parameter['symbol_id'] = context['parameters'][0]['symbol_id']
    elif fault == 'parameter_name_duplicate': parameter['name'] = context['parameters'][0]['name']
    elif fault == 'provider_version': metadata['version'] = 2
    elif fault == 'provider_admission': metadata['conditional_guard_before_source_read_admission'] = False
    elif fault == 'provider_generation': metadata['generation_snapshot_checked'] = False
    elif fault == 'provider_integer': metadata['original_integer_prework'] = ''
    elif fault == 'provider_guard': metadata['guard_arithmetic'] = ''
    elif fault == 'provider_zero': metadata['zero_trip_unit'] = ''
    elif fault == 'provider_parameter_order': metadata['parameters'].reverse()
    elif fault == 'model_symbol_bool': domain['parameters'][0]['symbol_id'] = True
    elif fault == 'path_symbol_bool': context['paths'][0]['original_units'][1] = True
    elif fault == 'unit_missing': team['units'][0].pop('section_directive_node_id')
    elif fault == 'resource_nontext': parameter['resource'] = []
    elif fault == 'read_guard': variant['source_reads']['read_admission_requires_same_conditional_guard'] = False
    reseal(report)
    if fault == 'read_context': variant['source_reads']['conditional_numerical_context_identity'] = 'f'*64
    if fault == 'effect_context': variant['physical_effects']['conditional_numerical_context_identity'] = 'f'*64
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)
