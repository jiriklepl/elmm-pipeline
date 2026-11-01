"""Receiver coordinates stay with issued source evidence across tool installs."""
from pathlib import Path
import json
import shutil
import sys

import pytest

from elmm_pipeline.build_receipts import file_identity
from elmm_pipeline.gpu_pressure_build import prepare_lazy_pressure_backend
from elmm_pipeline.gpu_pressure_lazy import prepare_lazy_poisfft_contract
from test_gpu_pressure_build import backend_request


def fixture(tmp_path, monkeypatch):
    original=tmp_path/'backend';original.mkdir()
    request=backend_request(original,Path(sys.executable))
    selected=prepare_lazy_pressure_backend(request,tmp_path/'prepared-backend',fortran=sys.executable)
    canonical=Path(__file__).resolve().parents[1]/'elmm_pipeline/gpu_pressure_sources/gpu_pressure_lazy_poisfft.f90'
    historical=tmp_path/'issued-original-receiver.f90';shutil.copyfile(canonical,historical)
    qualified={'kind':'explicit_lazy_cpu_poisfft_receiver_qualification','implementation':file_identity(historical)}
    qualification=tmp_path/'receiver-qualification.json';qualification.write_text(json.dumps(qualified))
    analysis={'source_inputs':{str(historical):file_identity(historical)['sha256']},
              'entries':[{'source':str(historical),'path':str(historical),'line_map':[4,8,19]}]}
    captured=[]
    def issuer(source_paths,document,destination,**kwargs):
        captured.append((source_paths,document,destination,kwargs))
        return {'proof':'fresh issuer must still run'}
    monkeypatch.setattr('elmm_pipeline.gpu_pressure_contract._prepare_poisfft_contract',issuer)
    kwargs={'backend':selected,'fortran':sys.executable,'numerical_qualification':file_identity(qualification),
            'opaque_qualification':file_identity(qualification),'workspace_cap_bytes':4194304}
    return canonical,historical,qualification,analysis,kwargs,captured


def test_issued_receiver_path_survives_tool_relocation_without_map_translation(tmp_path,monkeypatch):
    canonical,historical,qualification,analysis,kwargs,captured=fixture(tmp_path,monkeypatch)
    sources=[historical];before=json.dumps(analysis,sort_keys=True)
    result=prepare_lazy_poisfft_contract(sources,analysis,tmp_path/'result',**kwargs)
    assert result=={'proof':'fresh issuer must still run'} and len(captured)==1
    assert captured[0][0] is sources and captured[0][1] is analysis
    assert json.dumps(analysis,sort_keys=True)==before
    assert str(canonical) not in analysis['source_inputs']
    assert captured[0][3]['opaque_qualification']==file_identity(qualification)
    assert captured[0][3]['callee']=='elmm_gpu_pressure_lazy_poisfft::lazy_execute'


@pytest.mark.parametrize('change',['original_bytes','qualified_hash','missing_path','translated_path','bad_kind',
                                  'relative_path','extra_identity_field','qualification_bytes','backend_bytes','backend_source'])
def test_receiver_source_or_package_mismatch_stops_before_issuer(tmp_path,monkeypatch,change):
    canonical,historical,qualification,analysis,kwargs,captured=fixture(tmp_path,monkeypatch)
    q=json.loads(qualification.read_text())
    if change=='original_bytes':historical.write_text(historical.read_text()+'! changed original\n')
    elif change=='qualified_hash':q['implementation']['sha256']='0'*64
    elif change=='missing_path':analysis['source_inputs'].clear()
    elif change=='translated_path':
        other=tmp_path/'equal-bytes-different-source.f90';shutil.copyfile(historical,other)
        analysis['source_inputs']={str(other):file_identity(other)['sha256']}
    elif change=='bad_kind':q['kind']='unchanged_original_PoisFFT_receiver'
    elif change=='relative_path':q['implementation']['path']=historical.name
    elif change=='extra_identity_field':q['implementation']['bounds']=[1,2]
    elif change=='backend_bytes':Path(kwargs['backend']['manifest']).write_text('{}')
    elif change=='backend_source':
        backend=json.loads(Path(kwargs['backend']['manifest']).read_text())
        Path(backend['requirements']['cpu_fallback']['source']['path']).write_text('! changed packaged receiver\n')
    if change!='qualification_bytes':
        qualification.write_text(json.dumps(q));kwargs['opaque_qualification']=file_identity(qualification)
    else:qualification.write_text(qualification.read_text()+' ')
    with pytest.raises(ValueError):prepare_lazy_poisfft_contract([historical],analysis,tmp_path/'rejected',**kwargs)
    assert not captured and not (tmp_path/'rejected').exists()
