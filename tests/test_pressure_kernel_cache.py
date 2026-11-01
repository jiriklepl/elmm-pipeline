"""Bound launch policy; no CUDA, compiler invocation or application prewarming."""
from pathlib import Path
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from elmm_pipeline.build_receipts import file_identity
from elmm_pipeline.common import clean_environment
from elmm_pipeline.gpu_pressure_build import _json_identity
from elmm_pipeline.campaigns import pressure_cache as cache
from elmm_pipeline.campaigns import run as running
from elmm_pipeline.campaigns import profile as profiling
from elmm_pipeline.campaigns.identity import digest, write, tree_hashes
from elmm_pipeline.campaigns.prepare import check
from test_gpu_pressure_build import backend_request
from test_campaign_profile import fixture as profile_fixture

ENV = cache.CACHE_ENVIRONMENT_VARIABLE


def backend(tmp_path, *, change=None):
    directory = tmp_path / 'provider'
    directory.mkdir()
    request = backend_request(directory, Path(sys.executable))
    path = Path(request['manifest'])
    value = json.loads(path.read_text())
    install = directory / 'install'
    library = install / 'lib/libpoisson-solver.so'
    library.parent.mkdir(parents=True)
    shutil.copyfile(value['provider_library']['path'], library)
    identity = install / 'share/PoissonSolver/kernel-cache-identity.json'
    identity.parent.mkdir(parents=True)
    recipe = identity.with_name('kernel-cache-build-recipe.txt')
    recipe.write_bytes(b'fixture recipe bytes; no executable cache images\n')
    record = {'schema_version': 1, 'kind': 'poisson-nvrtc-cubin-cache', 'cache_format': 'POISSON_CUBIN_1',
              'build_key': digest(recipe), 'environment_variable': ENV,
              'max_image_bytes': 16 << 20, 'max_cache_bytes': 64 << 20,
              'nvrtc_builtins': file_identity(sys.executable), 'recipe_file': recipe.name}
    if change == 'unknown': record['extra'] = True
    if change == 'boolean': record['schema_version'] = True
    if change == 'build_key': record['build_key'] = '0' * 64
    if change == 'escape': record['recipe_file'] = '../escape'
    if change == 'builtins': record['nvrtc_builtins']['path'] = 'relative'
    if change == 'nul': record['nvrtc_builtins']['path'] = '/bad\0path'
    write(identity, record)
    if change == 'json': identity.write_text('{invalid')
    if change == 'legacy': identity.unlink(); recipe.unlink()
    value['objects'] = [str(library)]
    value['provider_library'] = file_identity(library)
    value['inputs'] = [file_identity(library)]
    if identity.exists() and change != 'unbound_identity': value['inputs'].append(file_identity(identity))
    if recipe.exists() and change != 'unbound_recipe': value['inputs'].append(file_identity(recipe))
    if change == 'symlink':
        actual = identity.with_name('other.json')
        identity.rename(actual)
        identity.symlink_to(actual.name)
        value['inputs'] = [file_identity(library), file_identity(identity), file_identity(recipe)]
    value['identity'] = _json_identity({k: v for k, v in value.items() if k != 'identity'})
    write(path, value)
    return {'manifest': str(path), 'sha256': digest(path)}, record


def campaign(root, request, state=None):
    root.mkdir(exist_ok=True)
    state = dict(state or {})
    adapter = root / 'adapter.json'
    write(adapter, {'pressure_backend': request})
    state.setdefault('modes', [{'name': 'candidate', 'binary': 'application'}])
    state['modes'][0]['adapter'] = adapter.name
    state.setdefault('cases', [])
    for case in state['cases']:
        case['config_sha256'] = tree_hashes(root / case['config'])
    state.update(status='prepared', files={'adapter.json': digest(adapter)},
                 external_files={request['manifest']: request['sha256']})
    write(root / 'campaign.json', state)
    write(root / 'identity.json', {'campaign_sha256': digest(root / 'campaign.json')})
    return state


def apply(root, state, mode=None):
    env = {ENV: '/inherited/forbidden'}
    result = cache.apply_pressure_cache(root, state, mode or state['modes'][0], env)
    return result, env


def test_clean_environment_removes_inherited_cache_without_mutating_parent(monkeypatch):
    monkeypatch.setenv(ENV, '/inherited/cache')
    assert ENV not in clean_environment(4)
    assert os.environ[ENV] == '/inherited/cache'
    record, env = apply(Path('/unused'), {}, {'name': 'native'})
    assert record['policy'] == 'disabled' and ENV not in env


def test_bound_identity_creates_private_persistent_mode_backend_directory(tmp_path, monkeypatch):
    request, identity = backend(tmp_path)
    root = tmp_path / 'campaign';state = campaign(root, request)
    monkeypatch.setenv('FC', '/untrusted/fortran')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('no subprocess/precompile is permitted'))
    first, env = apply(root, state)
    path = Path(env[ENV])
    assert path == root / '.poisson-kernel-cache/candidate' / request['sha256']
    assert first['build_key'] == identity['build_key'] and not first['directory_reused']
    for item in [path, path.parent, path.parent.parent]: assert stat.S_IMODE(item.stat().st_mode) == 0o700
    (path / 'retained.fixture').write_bytes(b'warmup payload')
    second, other = apply(root, state)
    assert second['directory_reused'] and other == env
    assert (path / 'retained.fixture').read_bytes() == b'warmup payload'
    assert check(root) == state  # Cache growth does not mutate frozen input trees.


def test_legacy_backend_still_uses_public_validator_and_disables_cache(tmp_path, monkeypatch):
    request, _ = backend(tmp_path, change='legacy');root=tmp_path/'campaign';state=campaign(root,request)
    original=cache.require_pressure_backend;calls=[]
    def validate(*args, **kwargs): calls.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(cache,'require_pressure_backend',validate)
    record,env=apply(root,state)
    assert len(calls)==1 and record['reason']=='legacy_backend_without_cache_identity'
    assert ENV not in env and not (root/'.poisson-kernel-cache').exists()


@pytest.mark.parametrize('change',['unknown','boolean','build_key','escape','builtins','nul','json','unbound_identity','unbound_recipe','symlink'])
def test_present_identity_must_be_well_formed_and_exactly_bound(tmp_path,change):
    request,_=backend(tmp_path,change=change);root=tmp_path/'campaign';state=campaign(root,request)
    with pytest.raises(ValueError): apply(root,state)
    assert not (root/'.poisson-kernel-cache').exists()


@pytest.mark.parametrize('change',['identity','recipe','backend','library','compiler'])
def test_changed_backend_inputs_never_fall_back_to_uncached_launch(tmp_path,change):
    request,_=backend(tmp_path);root=tmp_path/'campaign';state=campaign(root,request)
    manifest=json.loads(Path(request['manifest']).read_text())
    if change=='backend': target=Path(request['manifest'])
    elif change=='compiler':
        manifest['fortran_compiler']['sha256']='0'*64
        manifest['identity']=_json_identity({k:v for k,v in manifest.items() if k!='identity'})
        write(Path(request['manifest']),manifest);request['sha256']=digest(request['manifest']);state=campaign(root,request)
        target=None
    elif change=='library':target=Path(manifest['provider_library']['path'])
    else:target=next(Path(x['path']) for x in manifest['inputs'] if Path(x['path']).name==('kernel-cache-identity.json' if change=='identity' else 'kernel-cache-build-recipe.txt'))
    if target is not None:target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError):apply(root,state)
    assert not (root/'.poisson-kernel-cache').exists()


@pytest.mark.parametrize('change',['symlink','permissions','file','leaf_symlink','foreign_owner','root_permissions','frozen_overlap','bad_mode'])
def test_unsafe_or_frozen_cache_locations_reject_before_launch(tmp_path,monkeypatch,change):
    request,_=backend(tmp_path);root=tmp_path/'campaign';state=campaign(root,request);base=root/'.poisson-kernel-cache'
    if change=='symlink':base.symlink_to(tmp_path,target_is_directory=True)
    elif change=='permissions':base.mkdir(mode=0o755)
    elif change=='file':base.write_text('not a directory')
    elif change=='leaf_symlink':
        (base/'candidate').mkdir(parents=True,mode=0o700);base.chmod(0o700)
        (base/'candidate'/request['sha256']).symlink_to(tmp_path,target_is_directory=True)
    elif change=='foreign_owner':monkeypatch.setattr(cache.os,'geteuid',lambda:root.stat().st_uid+1)
    elif change=='root_permissions':root.chmod(0o770)
    elif change=='frozen_overlap':
        base.mkdir(mode=0o700);p=base/'protected';p.write_text('frozen')
        state['files'][str(p.relative_to(root))]=digest(p)
        write(root/'campaign.json',state);write(root/'identity.json',{'campaign_sha256':digest(root/'campaign.json')})
    elif change=='bad_mode':state['modes'][0]['name']='../escape';write(root/'campaign.json',state);write(root/'identity.json',{'campaign_sha256':digest(root/'campaign.json')})
    with pytest.raises(ValueError):apply(root,state)


def test_campaign_guard_precedes_cache_creation(tmp_path):
    request,_=backend(tmp_path);root=tmp_path/'campaign';state=campaign(root,request)
    (root/'campaign.json').write_text('{}')
    with pytest.raises(ValueError,match='campaign configuration changed'):apply(root,state)
    assert not (root/'.poisson-kernel-cache').exists()


def test_timed_process_populates_cache_and_next_measurement_reuses_it(tmp_path,monkeypatch):
    request,_=backend(tmp_path);root=tmp_path/'campaign'
    state=campaign(root,request,{'threads':4,'selected_gpu_uuid':'GPU-fixture','tools':{'time':{'path':'time'}}})
    (root/'config').mkdir();case={'name':'case','config':'config','reference':'reference.vtk','completed_steps':1,'configuration':{'runtime_arguments':[],'expected_max_steps':1}}
    monkeypatch.setattr(running,'available',lambda:4*2**30)
    monkeypatch.setattr(running,'read_vtk',lambda p:None);monkeypatch.setattr(running,'compare_vtk',lambda *a:{'passed':True});monkeypatch.setattr(running,'validate_log',lambda *a,**k:{})
    invocations=[]
    def execute(argv,**kwargs):
        path=Path(kwargs['env'][ENV]);marker=path/'compiled.fixture';invocations.append(marker.exists());marker.write_text('created inside application')
        Path(next(x.split('=',1)[1] for x in argv if x.startswith('--output='))).write_text('1024')
        return subprocess.CompletedProcess(argv,0,stdout='complete',stderr='')
    monkeypatch.setattr(running.subprocess,'run',execute)
    records=[running.run_one(root,state,case,state['modes'][0],root/name,traced=False) for name in ['warmup','measured']]
    assert invocations==[False,True]
    assert [r['pressure_kernel_cache']['directory_reused'] for r in records]==[False,True]
    assert all('wall_seconds' in r for r in records)


def test_profile_uses_same_cache_policy_and_records_it(tmp_path,monkeypatch):
    request,_=backend(tmp_path)
    root,output,profiler,_=profile_fixture(tmp_path,monkeypatch,profiler_mode='forward')
    state=campaign(root,request,profiling.check(root))
    monkeypatch.setattr(profiling,'check',lambda p:state)
    warm,_=apply(root,state)
    record=profiling.profile(root,output,case='case',mode='candidate',profiler=profiler,seconds=1)
    assert record['pressure_kernel_cache']['path']==warm['path']
    assert record['pressure_kernel_cache']['directory_reused']
    assert not record['timing_sample']


def test_unbound_adapter_cannot_select_a_cache(tmp_path):
    request,_=backend(tmp_path);root=tmp_path/'campaign';state=campaign(root,request)
    state['files'].clear()
    write(root/'campaign.json',state);write(root/'identity.json',{'campaign_sha256':digest(root/'campaign.json')})
    with pytest.raises(ValueError,match='not a frozen campaign input'):apply(root,state)
    assert not (root/'.poisson-kernel-cache').exists()


def test_another_mode_cannot_borrow_the_frozen_adapter_policy(tmp_path):
    request,_=backend(tmp_path);root=tmp_path/'campaign';state=campaign(root,request)
    with pytest.raises(ValueError,match='frozen adapter selection'):
        apply(root,state,{'name':'unissued-mode','adapter':'adapter.json'})


def test_campaign_symlink_is_not_a_mutable_cache_anchor(tmp_path):
    request,_=backend(tmp_path);root=tmp_path/'campaign';state=campaign(root,request)
    linked=tmp_path/'linked';linked.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match='without symlinks'):apply(linked,state)
    assert not (root/'.poisson-kernel-cache').exists()
