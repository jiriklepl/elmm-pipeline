import copy
import hashlib
import pytest
from elmm_pipeline.gpu_kernel_origins import kernel_source_origins, attribute_kernel


def fixture(tmp_path, kinds=('original_line_snapshot_team',) * 3):
    tasks=[];sources=[];hashes={}
    for number,kind in enumerate(kinds):
        entry='provider'+str(number);name=entry+'.cu';text='namespace { static __global__ void kernel_tile_private_0(int n) {} }'
        (tmp_path/name).write_text(text);digest=hashlib.sha256(text.encode()).hexdigest();hashes[name]=digest;sources.append({'path':name,'language':'cuda'})
        tasks.append({'procedure':'mod::lines','identity':str(number),'execution_binding':{'provider':{'entry':entry,'cuda_source_sha256':digest,'metadata':{'cuda_sha256':digest,'kind':kind}}}})
    name='entry.cu';text='namespace generated_kernels::fort_shared_ab12 { void entry() {} }';(tmp_path/name).write_text(text);hashes[name]=hashlib.sha256(text.encode()).hexdigest();sources.append({'path':name,'language':'cuda'})
    return {'scopes':[{'checked_execution_plan':{'entry':'mod::owner'},'source_device_tasks':tasks}],'build_sources':sources,'artifacts_sha256':hashes}


def test_same_procedure_collisions_retain_all_three_origins(tmp_path):
    rows=kernel_source_origins(fixture(tmp_path),tmp_path)
    result=attribute_kernel('<unnamed>::kernel_tile_private_0(int)',rows)
    assert result['status']=='attributed' and len(result['candidates'])==3
    assert {r['provider_entry'] for r in result['candidates']}=={'provider0','provider1','provider2'}
    assert result['family']['procedure']=='mod::lines'
    assert 'axis identity is not established' in result['granularity']


def test_numerical_namespace_and_anonymous_spelling(tmp_path):
    rows=kernel_source_origins(fixture(tmp_path),tmp_path)
    assert attribute_kernel('generated_kernels::fort_shared_ab12::kernel_0(double*)',rows)['family']['task_kind']=='numerical_component'
    assert len(attribute_kernel('(anonymous namespace)::kernel_tile_private_0(int)',rows)['candidates'])==3


@pytest.mark.parametrize('field,value',[('procedure','mod::other'),('task_kind','original_tile_private_team'),('owner','mod::other_owner')])
def test_unrelated_collision_remains_ambiguous(tmp_path,field,value):
    rows=kernel_source_origins(fixture(tmp_path),tmp_path)
    assert attribute_kernel('<unnamed>::kernel_tile_private_0(int)',rows)['status']=='attributed'
    next(r for r in rows if r['symbol_kind']=='anonymous')[field]=value
    result=attribute_kernel('<unnamed>::kernel_tile_private_0(int)',rows)
    assert result['status']=='ambiguous' and result['family'] is None and len(result['candidates'])==3


@pytest.mark.parametrize('name',['unknown()','<unnamed>::kernel_tile_private_cta_0(int)','generated_kernels::fort_shared_dead::kernel_0()'])
def test_unknown_kernel_rejected(tmp_path,name):
    with pytest.raises(ValueError,match='no exact'):
        attribute_kernel(name,kernel_source_origins(fixture(tmp_path),tmp_path))


@pytest.mark.parametrize('mutation',['source','artifact_digest','provider_digest','metadata_digest','missing_build','two_components','declarations','escape'])
def test_changed_or_unbound_sources_reject(tmp_path,mutation):
    manifest=fixture(tmp_path);provider=manifest['scopes'][0]['source_device_tasks'][0]['execution_binding']['provider']
    if mutation=='source':(tmp_path/'provider0.cu').write_text('changed')
    elif mutation=='artifact_digest':manifest['artifacts_sha256']['provider0.cu']='0'*64
    elif mutation=='provider_digest':provider['cuda_source_sha256']='0'*64
    elif mutation=='metadata_digest':provider['metadata']['cuda_sha256']='0'*64
    elif mutation=='missing_build':manifest['build_sources']=manifest['build_sources'][1:]
    elif mutation=='two_components':manifest['scopes'].append(copy.deepcopy(manifest['scopes'][0]))
    elif mutation=='declarations':
        p=tmp_path/'provider0.cu';p.write_text('// no declaration');h=hashlib.sha256(p.read_bytes()).hexdigest();manifest['artifacts_sha256'][p.name]=provider['cuda_source_sha256']=provider['metadata']['cuda_sha256']=h
    else:
        outside=tmp_path.parent/'escape.cu';outside.write_text('outside');manifest['build_sources'].append({'path':'../escape.cu','language':'cuda'})
    with pytest.raises(ValueError):kernel_source_origins(manifest,tmp_path)
