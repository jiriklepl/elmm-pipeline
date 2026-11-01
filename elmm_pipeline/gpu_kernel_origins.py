"""Diagnostic kernel-name origins from exact generated artifacts.

This supplies attribution, never execution/source authority. Anonymous symbols can
name several providers; retain all candidates and qualify only a common family.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re

_NAMESPACE = re.compile(r'namespace\s+generated_kernels::(fort_shared_[0-9a-f]+)\s*\{')
_KERNEL = re.compile(r'static\s+__global__\s+void\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(')

def kernel_source_origins(manifest, artifact_root):
    """Inventory one emitted component; every CUDA source must be hash bound."""
    scopes = manifest.get('scopes', [])
    if len(scopes) != 1:
        raise ValueError('kernel origins require one original component manifest')
    scope = scopes[0]
    owner = scope['checked_execution_plan']['entry']
    root = Path(artifact_root).resolve(strict=True)
    build = {row['path'] for row in manifest['build_sources'] if row['language'] == 'cuda'}
    providers = {}
    for task in scope.get('source_device_tasks', []):
        provider = task['execution_binding']['provider']
        source = provider['entry'] + '.cu'
        if source not in build or source in providers:
            raise ValueError('provider source is missing or ambiguous in build sources')
        providers[source] = (task, provider)
    rows = []
    for name in sorted(build):
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('CUDA source escapes artifact root')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest['artifacts_sha256'].get(name):
            raise ValueError('CUDA source differs from emitted artifact digest')
        text = data.decode('utf-8')
        base = {'owner': owner, 'source_path': name, 'source_sha256': digest}
        if name in providers:
            task, provider = providers[name]
            metadata = provider['metadata']
            if digest != provider['cuda_source_sha256'] or digest != metadata['cuda_sha256']:
                raise ValueError('provider CUDA identity differs from exact source')
            symbols = _KERNEL.findall(text)
            if not symbols or len(symbols) != len(set(symbols)):
                raise ValueError('provider kernel declarations unavailable or duplicated')
            for symbol in symbols:
                rows.append(base | {'symbol_kind': 'anonymous', 'symbol': symbol,
                    'procedure': task['procedure'], 'task_kind': metadata['kind'],
                    'provider_entry': provider['entry'], 'source_task_identity': task['identity']})
        else:
            for namespace in _NAMESPACE.findall(text):
                rows.append(base | {'symbol_kind': 'namespace', 'symbol': namespace,
                    'procedure': owner, 'task_kind': 'numerical_component',
                    'provider_entry': None, 'source_task_identity': None})
    return rows


def attribute_kernel(name, origins):
    """Return every possible origin; never pick one anonymous provider by name."""
    namespace = re.search(r'generated_kernels::(fort_shared_[0-9a-f]+)::', name)
    anonymous = re.match(r'(?:<unnamed>|\(anonymous namespace\))::([A-Za-z_][A-Za-z_0-9]*)\(', name)
    key = ('namespace', namespace[1]) if namespace else ('anonymous', anonymous[1]) if anonymous else None
    matches = [r for r in origins if key == (r['symbol_kind'], r['symbol'])]
    if not matches:
        raise ValueError('GPU kernel has no exact generated source origin: ' + name)
    families = {(r['owner'], r['procedure'], r['task_kind']) for r in matches}
    if len(families) != 1:
        return {'status': 'ambiguous', 'granularity': 'unrelated source families',
                'kernel': name, 'candidates': matches, 'family': None}
    owner, procedure, kind = next(iter(families))
    return {'status': 'attributed', 'granularity': 'exact generated namespace' if namespace else
            'source family; anonymous provider/axis identity is not established',
            'kernel': name, 'candidates': matches,
            'family': {'owner': owner, 'procedure': procedure, 'task_kind': kind}}
