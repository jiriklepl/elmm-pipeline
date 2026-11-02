"""An inline leaf label needs exact reached source and active-domain evidence."""
import json
from copy import deepcopy

import pytest

from elmm_pipeline.gpu_validation import gpu_activity, offload_decisions

OWNER = 'b5413957e566fceadb1acc1e2e778c46942ae201bda8be9ea543875d479726f0'
CALLER = 'a4368ce9e51c5c6887e8368d8224f17e8ca6b73667e08e87693a5e8bf3010b23'
SEGMENT = '7b59072fdd7ad2d5b122f06adf495ac4193799dd45d34bc975a05a0d6a4c36f0'
WORKER = '46be7cac28034e47f3db5edc59dc9514c9af7afe3db0b74678c90db5fc343ef5'
CUDA = 'e2ad9365a22a6fc00e5b800ce50e89ba894c1347f85b41ccf0be60a9ad97dd67'
NUMERICAL = '4239a7e1a9ea33e7c5b5cbfa1ddee89a52778795ceac0b3c4687cafe690e2e3d'
SOURCE = '88519c6779e682b61e985d2e3bdf8053ffe2ac357ce39f61eccf993eab7351e0'
ENTRY = 'fort_inline_88519c6779e6::region'
PROCEDURE = 'subgrid::sgs_sigma_uniform'
ORIGINAL = '/tmp/fortran-cpu-pressure-source-current-01/application/elmm/src/subgrid.f90'
ORIGINAL_SHA = 'f89691ebf9bbf2033607f60a1870aada38056215c846ae46a9b521de046247bf'


def fixture():
    # Exact first short06 hidden fallback identities and original749:773 span.
    records = [
        {'id': OWNER, 'kind': 'owner', 'procedure': PROCEDURE, 'source': ORIGINAL,
         'source_sha256': ORIGINAL_SHA, 'lifetime': 'invocation'},
        {'id': CALLER, 'kind': 'procedure', 'procedure': PROCEDURE, 'source': ORIGINAL, 'source_sha256': ORIGINAL_SHA},
        {'id': SEGMENT, 'kind': 'segment', 'procedure': PROCEDURE, 'source': ORIGINAL,
         'source_sha256': ORIGINAL_SHA, 'first_line': 749, 'last_line': 773},
        {'id': WORKER, 'kind': 'procedure', 'procedure': ENTRY,
         'source': 'inline:' + SOURCE, 'numerical_identity': NUMERICAL},
        {'id': CUDA, 'kind': 'implementation', 'procedure_id': WORKER, 'procedure': ENTRY,
         'source': 'inline:' + SOURCE, 'numerical_identity': NUMERICAL, 'region': 0, 'backend': 'cuda'},
    ]
    manifest = {'schema_version': 1, 'scope_count': 1, 'runtime_provenance': {'schema_version': 1, 'records': records},
                'scopes': [{'mode': 'sections', 'gpu_leaves': [PROCEDURE + '#region3'],
                    'checked_execution_plan': {'entry': PROCEDURE, 'procedures': [{'procedure': PROCEDURE}]}}],
                'inline_numerical_regions': {'regions': [{'entry': ENTRY, 'source_identity': SOURCE,
                    'first_line': 749, 'last_line': 773, 'completion': {'procedure': PROCEDURE}}]}}
    unknown = dict.fromkeys(('owner', 'procedure', 'segment', 'operation', 'implementation', 'boundary'), 'unknown')
    create = {'schema_version': 1, 'event': 'domain_create', 'domain': 1, 'context': 0,
              'physical_context': 1, 'last_context': 0, 'device': 0, 'failed': False, **unknown}
    attach = {**create, 'event': 'domain_attach', 'context': 2}
    position = ('FORT_SCOPED position bytes=0 provenance_version=1 context=2 domain_version=1 domain=1 domain_role=execution '
                f'owner={OWNER} procedure={CALLER} segment={SEGMENT} operation=unknown implementation={CUDA} boundary=unknown')
    decision = 'FORT_OFFLOAD entry=region mode=native-scoped-resource gpu_units=0 cpu_units=1 unit_kind=regions'
    lines = ['FORT_SCOPED domain ' + json.dumps(row) for row in (create, attach)] + [position, decision]
    return manifest, lines


@pytest.mark.parametrize('reason', ['resource', 'boundary'])
@pytest.mark.parametrize('composed', [False, True])
def test_saved_first_inline_fallback_retains_raw_and_exact_qualified_attribution(reason, composed):
    manifest, lines = fixture()
    before = deepcopy(manifest)
    lines[-1] = lines[-1].replace('native-scoped-resource', 'native-scoped-' + reason)
    supplied = ({'schema_version': 1, 'scope_count': 1, 'scopes': manifest['scopes'],
                 'runtime_provenance': manifest['runtime_provenance'], 'component_manifests': [manifest]}
                if composed else manifest)
    result = gpu_activity('\n'.join(lines), 'local', [], scope_manifest=supplied)
    decisions = result['offload_decisions']
    assert not result['verified'] and result['kernel_launches'] == decisions['gpu_units'] == 0
    assert decisions['calls'] == decisions['modes']['native'] == 1
    assert set(decisions['entries']) == {ENTRY}
    row, = decisions['authenticated_inline_fallbacks']
    assert row == {'entry': ENTRY, 'raw_entry': 'region', 'owner': OWNER, 'procedure': CALLER,
                   'segment': SEGMENT, 'implementation': CUDA, 'worker_procedure': WORKER,
                   'numerical_identity': NUMERICAL, 'source_identity': SOURCE,
                   'first_line': 749, 'last_line': 773, 'log_line': 4, 'context': 2, 'domain': 1,
                   'mode': 'native-scoped-' + reason, 'gpu_units': 0, 'cpu_units': 1,
                   'unit_kind': 'regions', 'gpu_execution_confirmed': False}
    assert manifest == before


@pytest.mark.parametrize('change', [
    'missing_position', 'interposed_line', 'missing_domain', 'detached', 'foreign_context',
    'foreign_domain', 'version', 'extra_position_field', 'duplicate_position_field',
    'foreign_implementation', 'foreign_owner', 'foreign_procedure', 'foreign_segment',
    'wrong_backend', 'wrong_numerical', 'wrong_worker', 'wrong_source',
    'wrong_original_procedure', 'outside_span', 'cross_scope', 'missing_inline',
    'gpu_count', 'cpu_count', 'unit_kind', 'mode', 'second_decision',
    'malformed_mode', 'negative_gpu_count', 'malformed_cpu_count', 'negative_context',
    'malformed_domain', 'implementation_kind', 'lifecycle_context', 'lifecycle_failed',
])
def test_unknown_leaf_never_acquires_global_or_foreign_authority(change):
    manifest, lines = fixture()
    records = manifest['runtime_provenance']['records']
    region = manifest['inline_numerical_regions']['regions'][0]
    if change == 'missing_position':
        del lines[-2]
    elif change == 'interposed_line':
        lines.insert(-1, 'an unrelated line')
    elif change == 'missing_domain':
        del lines[:2]
    elif change == 'detached':
        attach = json.loads(lines[1].removeprefix('FORT_SCOPED domain '))
        lines.insert(-2, 'FORT_SCOPED domain ' + json.dumps({**attach, 'event': 'domain_detach', 'last_context': 2}))
    elif change in {'foreign_context', 'foreign_domain', 'version'}:
        old, new = {'foreign_context': ('context=2', 'context=3'),
                    'foreign_domain': ('domain=1', 'domain=5'), 'version': ('provenance_version=1', 'provenance_version=2')}[change]
        lines[-2] = lines[-2].replace(old, new)
    elif change in {'extra_position_field', 'duplicate_position_field'}:
        lines[-2] += ' ' + ('extra=1' if change == 'extra_position_field' else 'bytes=0')
    elif change in {'negative_context', 'malformed_domain'}:
        lines[-2] = lines[-2].replace('context=2', 'context=-1') if change == 'negative_context' else lines[-2].replace('domain=1', 'domain=bogus')
    elif change in {'lifecycle_context', 'lifecycle_failed'}:
        row = json.loads(lines[1].removeprefix('FORT_SCOPED domain '))
        row['context' if change == 'lifecycle_context' else 'failed'] = -1 if change == 'lifecycle_context' else 'false'
        lines[1] = 'FORT_SCOPED domain ' + json.dumps(row)
    elif change.startswith('foreign_'):
        key = change.removeprefix('foreign_')
        previous = {'implementation': CUDA, 'owner': OWNER, 'procedure': CALLER, 'segment': SEGMENT}[key]
        lines[-2] = lines[-2].replace(key + '=' + previous, key + '=' + 'f' * 64)
    elif change == 'wrong_backend':
        records[-1]['backend'] = 'generated_cpu'
    elif change == 'implementation_kind':
        records[-1]['kind'] = 'procedure'
    elif change == 'wrong_numerical':
        records[-1]['numerical_identity'] = 'f' * 64
    elif change == 'wrong_worker':
        records[-1]['procedure_id'] = CALLER
    elif change == 'wrong_source':
        records[-1]['source'] = 'inline:' + 'f' * 64
    elif change == 'wrong_original_procedure':
        region['completion']['procedure'] = 'unrelated::procedure'
    elif change == 'outside_span':
        region['last_line'] = 774
    elif change == 'cross_scope':
        manifest['scopes'][0]['checked_execution_plan']['entry'] = 'foreign::root'
    elif change == 'missing_inline':
        manifest['inline_numerical_regions']['regions'] = []
    elif change in {'gpu_count', 'cpu_count', 'unit_kind', 'mode', 'malformed_mode', 'negative_gpu_count', 'malformed_cpu_count'}:
        old, new = {'gpu_count': ('gpu_units=0', 'gpu_units=1'), 'cpu_count': ('cpu_units=1', 'cpu_units=2'),
                    'unit_kind': ('unit_kind=regions', 'unit_kind=slabs'),
                    'mode': ('mode=native-scoped-resource', 'mode=scoped-scheduled'),
                    'malformed_mode': ('mode=native-scoped-resource', 'mode=bogus'),
                    'negative_gpu_count': ('gpu_units=0', 'gpu_units=-1'),
                    'malformed_cpu_count': ('cpu_units=1', 'cpu_units=bogus')}[change]
        lines[-1] = lines[-1].replace(old, new)
    else:
        lines.append(lines[-1])
    with pytest.raises(ValueError):
        offload_decisions('\n'.join(lines), [], scope_manifest=manifest)


def test_foreign_component_record_does_not_authorize_same_label_in_current_owner():
    manifest, lines = fixture()
    foreign = deepcopy(manifest)
    foreign['scopes'][0]['checked_execution_plan']['entry'] = 'foreign::root'
    manifest['inline_numerical_regions']['regions'] = []
    with pytest.raises(ValueError):
        offload_decisions('\n'.join(lines), [], scope_manifest={
            'component_manifests': [manifest, foreign], 'runtime_provenance': manifest['runtime_provenance']})


def test_unmatched_known_legacy_policy_row_keeps_its_existing_behavior():
    assert offload_decisions('FORT_OFFLOAD entry=known mode=bogus gpu_units=-1 cpu_units=1',
                             [{'entry': 'known'}])['calls'] == 0


def borrowed_fixture():
    manifest, lines = fixture()
    root, owner = 'root::entry', 'helper::owner'
    records = manifest['runtime_provenance']['records']
    records[0].update(procedure=owner, lifetime='borrowed helper storage epoch',
                      source='/app/helper.f90', source_sha256='a' * 64)
    def member(name, source, digest, graph, calls=()):
        return {'procedure': name, 'source': source, 'source_sha256': digest,
                'structured_summary_identity': graph, 'calls': list(calls), 'units': []}
    def call(parent, line):
        return {'caller': parent, 'first_line': line, 'last_line': line,
                'handle_bindings': {'10': 20}, 'resource_mappings': {'argument::x': 'root::data'}}
    members = [member(root, '/app/root.f90', 'b' * 64, '1' * 64),
               member(owner, '/app/helper.f90', 'a' * 64, '2' * 64, [call(root, 5)]),
               member(PROCEDURE, ORIGINAL, ORIGINAL_SHA, '3' * 64, [call(owner, 11)])]
    scope = manifest['scopes'][0]
    scope['checked_execution_plan'] = {'entry': root, 'procedures': members,
        'sources': {row['source']: row['source_sha256'] for row in members}}
    scope['module_coordinators'] = [{'procedure': row['procedure'], 'parameterized_handles': True,
                                    'calls': deepcopy(row['calls'])} for row in members[1:]]
    return manifest, lines


def test_borrowed_helper_owner_needs_two_exact_checked_companion_call_routes():
    manifest, lines = borrowed_fixture()
    result = offload_decisions('\n'.join(lines), [], scope_manifest=manifest)
    row, = result['authenticated_inline_fallbacks']
    route = row['borrowed_owner_route']
    assert route['root_to_owner'][0]['caller'] == 'root::entry'
    assert route['root_to_owner'][0]['callee'] == route['owner_to_caller'][0]['caller'] == 'helper::owner'
    edge, = route['owner_to_caller']
    assert edge['callee'] == PROCEDURE
    assert edge['resource_mappings'] == {'argument::x': 'root::data'}
    assert edge['handle_bindings'] == {'10': 20}
    assert len(edge['checked_call_identity']) == len(edge['coordinator_identity']) == 64
    assert result['gpu_units'] == 0 and row['gpu_execution_confirmed'] is False


def test_unrelated_unmatched_call_is_not_used_or_required_for_owner_route():
    manifest, lines = borrowed_fixture()
    members = manifest['scopes'][0]['checked_execution_plan']['procedures']
    members.append({'procedure': 'unrelated::callee', 'source': '/app/unrelated.f90',
        'source_sha256': 'c' * 64, 'structured_summary_identity': '4' * 64,
        'units': [], 'calls': [{'caller': 'root::entry', 'first_line': 99, 'last_line': 99,
            'handle_bindings': {'1': 2}, 'resource_mappings': {'argument::x': 'unrelated::data'}}]})
    row, = offload_decisions('\n'.join(lines), [], scope_manifest=manifest)['authenticated_inline_fallbacks']
    assert all(edge['callee'] != 'unrelated::callee' for path in ('root_to_owner', 'owner_to_caller')
               for edge in row['borrowed_owner_route'][path])


@pytest.mark.parametrize('change', [
    'lifetime', 'owner_source', 'caller_source', 'owner_graph', 'root_graph', 'caller_graph', 'root_source',
    'missing_root_path', 'missing_worker_path', 'missing_coordinator', 'foreign_coordinator', 'not_parameterized',
    'coordinator_span', 'coordinator_handles', 'coordinator_mappings', 'checked_span', 'checked_handles',
    'checked_mappings', 'duplicate_procedure', 'duplicate_coordinator', 'bad_handle_key', 'bad_handle_value',
])
def test_borrowed_owner_cannot_borrow_component_membership_or_another_call(change):
    manifest, lines = borrowed_fixture()
    scope = manifest['scopes'][0]
    members = scope['checked_execution_plan']['procedures']
    coordinator = scope['module_coordinators'][1]
    checked = members[2]['calls'][0]
    published = coordinator['calls'][0]
    if change == 'lifetime':
        manifest['runtime_provenance']['records'][0]['lifetime'] = 'invocation'
    elif change == 'owner_source':
        members[1]['source'] = '/app/foreign.f90'
    elif change == 'caller_source':
        members[2]['source_sha256'] = 'f' * 64
    elif change == 'root_source':
        members[0]['source_sha256'] = 'f' * 64
    elif change in {'owner_graph', 'root_graph', 'caller_graph'}:
        index = {'owner_graph': 1, 'root_graph': 0, 'caller_graph': 2}[change]
        members[index]['structured_summary_identity'] = 'unpublished-graph'
    elif change == 'missing_root_path':
        members[1]['calls'] = []
    elif change == 'missing_worker_path':
        members[2]['calls'] = []
    elif change == 'missing_coordinator':
        scope['module_coordinators'] = scope['module_coordinators'][:1]
    elif change == 'foreign_coordinator':
        coordinator['procedure'] = 'foreign::callee'
    elif change == 'not_parameterized':
        coordinator['parameterized_handles'] = False
    elif change.endswith('_span'):
        (published if change.startswith('coordinator') else checked)['first_line'] = 12
    elif change.endswith('_handles'):
        (published if change.startswith('coordinator') else checked)['handle_bindings']['10'] = 21
    elif change.endswith('_mappings'):
        (published if change.startswith('coordinator') else checked)['resource_mappings']['argument::x'] = 'foreign::data'
    elif change == 'duplicate_procedure':
        members.append(deepcopy(members[1]))
    elif change == 'duplicate_coordinator':
        scope['module_coordinators'].append(deepcopy(coordinator))
    else:
        handles = {'bad': 20} if change == 'bad_handle_key' else {'10': -1}
        checked['handle_bindings'] = handles
        published['handle_bindings'] = deepcopy(handles)
    with pytest.raises(ValueError):
        offload_decisions('\n'.join(lines), [], scope_manifest=manifest)
