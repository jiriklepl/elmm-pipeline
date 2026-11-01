"""Resident producer -> pinned Poisson operator -> exact faces -> consumer."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from elmm_pipeline.gpu_pressure import PressureProblem, prepare_adapter

CLIENT = r'''
#include "gpu_pressure.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void pressure_reference(const int *,const double *,const int *,int,void *,void *);
extern "C" int pressure_create_from_fortran(uint64_t,const elmm_gpu_pressure_config_v1 *,elmm_gpu_pressure_handle_v1 **);
extern "C" int pressure_solve_from_fortran(uint64_t,elmm_gpu_pressure_handle_v1 *,uint64_t,uint64_t,fort_scope_external_report_v1 *);
extern "C" int pressure_halo_from_fortran(uint64_t,elmm_gpu_pressure_handle_v1 *,uint64_t,fort_scope_external_report_v1 *);
extern "C" int pressure_destroy_from_fortran(elmm_gpu_pressure_handle_v1 **);
void rt(int status) {
    if(status) { std::fprintf(stderr,"runtime %d %s / adapter %s\n",status,fort_scope_error(),elmm_gpu_pressure_error_v1());std::abort(); }
}
void cu(cudaError_t status) { if(status!=cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(status));std::abort(); } }
size_t count(const int *n,int g) { return size_t(n[0]+2*g)*(n[1]+2*g)*(n[2]+2*g); }
size_t offset(const int *n,int g,int i,int j,int k) { return i+g+size_t(n[0]+2*g)*(j+g+size_t(n[1]+2*g)*(k+g)); }
struct Shape { int n[3];double coefficient; };
__global__ void producer(const double *a,double *rhs,Shape s) {
    for(int q=int(threadIdx.x);q<s.n[0]*s.n[1]*s.n[2];q+=int(blockDim.x)) {
        const int i=q%s.n[0],j=(q/s.n[0])%s.n[1],k=q/(s.n[0]*s.n[1]);
        rhs[(i+1)+(s.n[0]+2)*((j+1)+(s.n[1]+2)*(k+1))]=2*a[q];
    }
}
__global__ void consumer(const double *phi,double *out,Shape s) {
    const int nx=s.n[0]+4,ny=s.n[1]+4,nz=s.n[2]+4;
    for(int q=int(threadIdx.x);q<nx*ny*nz;q+=int(blockDim.x)) {
        const int i=q%nx,j=(q/nx)%ny,k=q/(nx*ny);
        const int outside=int(i<2 || i>=nx-2)+int(j<2 || j>=ny-2)+int(k<2 || k>=nz-2);
        if(outside<=1) out[q]=phi[q]+s.coefficient;
    }
}
int produce(void *data,const fort_scope_external_view_v1 *v,size_t n,void *stream,int *started) {
    assert(n==2);*started=1;producer<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const double *>(v[0].device),static_cast<double *>(v[1].device),*static_cast<Shape *>(data));
    return cudaGetLastError()==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
int consume(void *data,const fort_scope_external_view_v1 *v,size_t n,void *stream,int *started) {
    assert(n==2);*started=1;consumer<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const double *>(v[0].device),static_cast<double *>(v[1].device),*static_cast<Shape *>(data));
    return cudaGetLastError()==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
fort_scope_stats stats(fort_scope_domain_t domain) { fort_scope_stats s{};rt(fort_scope_domain_stats_get_v1(domain,&s));return s; }
int main(int argc,char **argv) {
    assert(argc==2 || argc==3);const bool hidden=argc==3;
    const int order=std::atoi(argv[1]);assert(order==2 || order==4);
    const int n[3]={order==2?8:7,order==2?6:5,order==2?10:9};const size_t active=count(n,0);
    const double sentinel=-0x1.23456789abcp+10;
    std::vector<double> a(active),rhs(count(n,1),sentinel),phi(count(n,2),sentinel),out(phi.size(),sentinel);
    double mean=0;for(size_t i=0;i<active;++i) { a[i]=std::sin(.19*i)+.01*double(i%13);mean+=a[i]; }
    mean/=active;for(auto &value:a) value-=mean;const auto original_a=a;
    auto expected_rhs=rhs;for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i)
        expected_rhs[offset(n,1,i,j,k)]=2*a[i+size_t(n[0])*(j+size_t(n[1])*k)];
    auto expected_phi=phi,reference_rhs=expected_rhs;const double lens[3]={2.75,1.25,3.5};const int bcs[6]={0,0,0,0,4,4};
    pressure_reference(n,lens,bcs,order,expected_phi.data(),reference_rhs.data());assert(reference_rhs==expected_rhs);
    fort_scope_domain_t domain=0;fort_scope_t context=0;rt(fort_scope_domain_create_v1(0,&domain));rt(fort_scope_create_in_domain_v1(domain,&context));
    rt(fort_scope_set_device_budget(context,24ULL<<20));
    elmm_gpu_pressure_config_v1 config{};config.version=1;config.solver.version=1;config.solver.dimensions=3;
    config.solver.approximation=order;config.solver.device_budget_bytes=4ULL<<20;
    std::copy(n,n+3,config.solver.sizes);std::copy(lens,lens+3,config.solver.lengths);std::copy(bcs,bcs+6,config.solver.boundary_conditions);
    std::fill_n(config.phi_lower_bounds,3,-1);std::fill_n(config.rhs_lower_bounds,3,0);
    elmm_gpu_pressure_handle_v1 *provider=nullptr;
    const int created=pressure_create_from_fortran(domain,&config,&provider);
    if(hidden) {
        assert(created==FORT_SCOPE_RESOURCE && !provider);
        const auto empty=stats(domain);
        assert(!empty.upload_bytes && !empty.download_bytes && !empty.allocations && !empty.launches);
        rt(fort_scope_close(context));rt(fort_scope_domain_close_v1(domain));
        // Exact original CPU operator remains usable after safe preflight rejection.
        rhs=expected_rhs;pressure_reference(n,lens,bcs,order,phi.data(),rhs.data());
        assert(a==original_a && rhs==expected_rhs);
        for(size_t i=0;i<phi.size();++i) assert(std::abs(phi[i]-expected_phi[i])<=2e-10+2e-10*std::abs(expected_phi[i]));
        std::printf("PRESSURE_HIDDEN_NATIVE_OK FD%d\n",order);return 0;
    }
    rt(created);assert(provider);
    size_t ae[3],re[3],pe[3];for(int i=0;i<3;++i) { ae[i]=n[i];re[i]=n[i]+2;pe[i]=n[i]+4; }
    int64_t al[3]={-4,9,0},rl[3]={0,0,0},pl[3]={-1,-1,-1};fort_buffer_t buffers[4]={};
    auto register_all=[&] {
        double *pointers[4]={a.data(),rhs.data(),phi.data(),out.data()};
        for(int i=0;i<4;++i) {
            fort_scope_layout layout{3,FORT_SCOPE_REAL64,8,pointers[i],i==0?ae:(i==1?re:pe),i==0?al:(i==1?rl:pl),17};
            fort_buffer_t handle=0;rt(fort_scope_register(context,i+1,17,&layout,1,&handle));
            if(buffers[i]) assert(buffers[i]==handle);else buffers[i]=handle;
        }
    };
    register_all();Shape shape{{n[0],n[1],n[2]},0};
    size_t rlo[3]={1,1,1},rhi[3];for(int i=0;i<3;++i) rhi[i]=n[i]+1;fort_scope_section rbox{rlo,rhi};
    fort_scope_access all_read{FORT_SCOPE_READ_ALL,0,nullptr,0,nullptr,0,nullptr};
    fort_scope_access overwrite{0,0,nullptr,1,&rbox,1,&rbox};
    fort_scope_external_binding_v1 pb[2]={{buffers[0],FORT_SCOPE_REAL64,3,8,all_read},{buffers[1],FORT_SCOPE_REAL64,3,8,overwrite}};
    fort_scope_external_call_v1 call{1,0,pb,2,0,0};fort_scope_external_report_v1 report{};
    rt(fort_scope_external_execute_v1(context,&call,produce,&shape,&report));assert(report.started && report.committed);
    rt(fort_scope_close(context));assert(stats(domain).upload_bytes==active*8 && !stats(domain).download_bytes);
    assert(rhs==std::vector<double>(rhs.size(),sentinel));
    rt(fort_scope_create_in_domain_v1(domain,&context));register_all();
    // The readonly host mirror remains valid while the device copy persists.
    rt(fort_scope_host_begin(context,buffers[0],&all_read));shape.coefficient=a.front()+a.back();rt(fort_scope_host_end(context,buffers[0]));
    assert(a==original_a && !stats(domain).download_bytes);
    rt(pressure_solve_from_fortran(context,provider,buffers[2],buffers[1],&report));assert(report.started && report.committed);
    assert(phi==std::vector<double>(phi.size(),sentinel));
    rt(pressure_halo_from_fortran(context,provider,buffers[2],&report));assert(report.started && report.committed);
    rt(fort_scope_close(context));
    rt(fort_scope_create_in_domain_v1(domain,&context));register_all();
    size_t lower[7][3],upper[7][3];fort_scope_section regions[7];
    for(int q=0;q<7;++q) { for(int i=0;i<3;++i) { lower[q][i]=2;upper[q][i]=n[i]+2; } }
    for(int axis=0;axis<3;++axis) for(int side=0;side<2;++side) {
        int q=1+2*axis+side;lower[q][axis]=side?n[axis]+2:0;upper[q][axis]=lower[q][axis]+2;
    }
    for(int q=0;q<7;++q) regions[q]={lower[q],upper[q]};
    fort_scope_access read_phi{0,7,regions,0,nullptr,0,nullptr},write_out{0,0,nullptr,7,regions,7,regions};
    fort_scope_external_binding_v1 cb[2]={{buffers[2],FORT_SCOPE_REAL64,3,8,read_phi},{buffers[3],FORT_SCOPE_REAL64,3,8,write_out}};
    call={1,0,cb,2,0,0};rt(fort_scope_external_execute_v1(context,&call,consume,&shape,&report));assert(report.started && report.committed);
    rt(fort_scope_close(context));
    const auto before=stats(domain);assert(before.upload_bytes==active*8 && !before.download_bytes && before.allocations==4);
    assert(fort_scope_domain_close_v1(domain)==FORT_SCOPE_STATE); // Outstanding provider lease cannot vanish.
    rt(pressure_destroy_from_fortran(&provider));assert(!provider);rt(fort_scope_domain_close_v1(domain));
    assert(a==original_a && rhs==expected_rhs);
    double maximum=0;for(int k=-2;k<n[2]+2;++k) for(int j=-2;j<n[1]+2;++j) for(int i=-2;i<n[0]+2;++i) {
        const auto q=offset(n,2,i,j,k);const int outside=int(i<0 || i>=n[0])+int(j<0 || j>=n[1])+int(k<0 || k>=n[2]);
        maximum=std::max(maximum,std::abs(phi[q]-expected_phi[q]));
        assert(std::abs(phi[q]-expected_phi[q])<=2e-10+2e-10*std::abs(expected_phi[q]));
        const double expected=outside<=1?expected_phi[q]+shape.coefficient:sentinel;
        assert(std::abs(out[q]-expected)<=2e-10+2e-10*std::abs(expected));
        if(outside>1) assert(phi[q]==sentinel && out[q]==sentinel);
    }
    std::printf("PRESSURE_DOMAIN_CHAIN_OK FD%d max_error=%.17g upload_bytes=%llu intermediate_downloads=0 contexts=3\n",order,maximum,(unsigned long long)before.upload_bytes);
}
'''


def _required(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to an authenticated pressure qualification prerequisite")
    path = Path(value).resolve()
    assert path.is_dir(), path
    return path


@pytest.fixture(scope="module")
def pressure_executable(tmp_path_factory):
    solver = _required("ELMM_PRESSURE_SOLVER_BUILD")
    solver_source = _required("ELMM_PRESSURE_SOLVER_SOURCE")
    native = _required("ELMM_PRESSURE_NATIVE_BUILD")
    app = _required("ELMM_PRESSURE_APPLICATION")
    runtime = _required("ELMM_PRESSURE_RUNTIME")
    fftw = _required("ELMM_PRESSURE_FFTW_DIR")
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    assert nvcc and host and fc
    directory = tmp_path_factory.mktemp("pressure_adapter")
    metadata = prepare_adapter(PressureProblem((7, 5, 9), (2.75, 1.25, 3.5), (0, 0, 0, 0, 4, 4), 4,
                                               4 << 20), directory / "sources")
    sources = directory / "sources"
    boundary = app / "elmm/src/boundaries.f90"
    parameters = (app / "elmm/src/parameters.f90").read_text()
    assert "We = 1, Ea = 2, So = 3, No = 4, Bo = 5, To = 6" in parameters
    assert "BC_PERIODIC = 3, BC_DIRICHLET = 4, BC_NEUMANN = 5" in parameters
    original = boundary.read_text()
    start = original.index("  subroutine Bound_Phi(Phi)")
    end = original.index("  end subroutine Bound_Phi", start) + len("  end subroutine Bound_Phi")
    exact_body = original[start:end]
    reference = directory / "reference.f90"
    reference.write_text('''module exact_boundary_reference
use iso_c_binding
implicit none
integer, parameter :: knd=c_double, We=1,Ea=2,So=3,No=4,Bo=5,To=6
integer, parameter :: BC_PERIODIC=3,BC_DIRICHLET=4,BC_MPI_BOUNDS_MIN=1000,BC_MPI_BOUNDS_MAX=1010
integer :: Prnx,Prny,Prnz,PrBtype(6)
contains
''' + exact_body + '''
subroutine pressure_reference(ns,lens,bcs,approximation,phi,rhs) bind(C)
use PoisFFT, only: PoisFFT_Solver3D_DP,Execute,Finalize
integer(c_int), intent(in) :: ns(3),bcs(6)
real(c_double), intent(in) :: lens(3)
integer(c_int),value :: approximation
type(c_ptr),value :: phi,rhs
real(c_double),pointer :: p(:,:,:),r(:,:,:)
type(PoisFFT_Solver3D_DP) :: solver
call c_f_pointer(phi,p,ns+4)
call c_f_pointer(rhs,r,ns+2)
solver=PoisFFT_Solver3D_DP(ns,lens,bcs,approximation)
call Execute(solver,p,r)
call Finalize(solver)
Prnx=ns(1);Prny=ns(2);Prnz=ns(3);PrBtype=merge(3,5,bcs==0)
call Bound_Phi(p)
end subroutine
end module
''')
    client = directory / "client.cu"
    client.write_text(CLIENT)
    commands = []

    def run(argv, label):
        commands.append(argv)
        result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=120, check=False)
        (directory / (label + ".stdout")).write_text(result.stdout)
        (directory / (label + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr

    objects = []
    for name, path, language in [
        ("runtime", runtime / "scoped_runtime.cu", "cuda"),
        ("bridge", sources / "gpu_pressure.cpp", "cxx"),
        ("halo", sources / "gpu_pressure_halo.cu", "cuda"),
        ("client", client, "cuda"),
    ]:
        target = directory / (name + ".o")
        compiler = [nvcc, "-ccbin", host, "-arch=sm_86", "--fmad=false"] if language == "cuda" else [host]
        run([*compiler, "-std=c++17", "-fPIC" if language == "cxx" else "-Xcompiler=-fPIC",
             "-I", str(runtime), "-I", str(solver_source / "include"), "-I", str(sources),
             "-I", str(Path(nvcc).resolve().parent.parent / "include"),
             "-c", str(path), "-o", str(target)], name)
        objects.append(str(target))
    run([fc, "-cpp", "-ffree-line-length-none", "-fopenmp", "-I", str(native), "-c", str(reference),
         "-o", str(directory / "reference.o")], "reference")
    # Compile the real BIND(C) module against the exact current runtime and provider modules.
    run([fc, "-ffree-line-length-none", "-c", str(runtime / "scoped_memory.f90"), "-o", str(directory / "memory.o")], "fortran-memory")
    run([fc, "-I", str(solver / "modules"), "-I", str(directory), "-c", str(sources / "gpu_pressure.f90"),
         "-o", str(directory / "fortran-api.o")], "fortran-api")
    calls = directory / "fortran_calls.f90"
    calls.write_text("""module tested_pressure_calls
use iso_c_binding
use elmm_gpu_pressure
implicit none
contains
function pressure_create_from_fortran(domain,config,handle) bind(C) result(status)
integer(c_int64_t),value :: domain
type(elmm_gpu_pressure_config_v1),intent(in) :: config
type(c_ptr),intent(out) :: handle
integer(c_int) :: status
status=elmm_gpu_pressure_create_v1(domain,config,handle)
end function
function pressure_solve_from_fortran(context,handle,phi,rhs,report) bind(C) result(status)
integer(c_int64_t),value :: context,phi,rhs
type(c_ptr),value :: handle
type(fort_scope_external_report_v1),intent(out) :: report
integer(c_int) :: status
status=elmm_gpu_pressure_solve_v1(context,handle,phi,rhs,report)
end function
function pressure_halo_from_fortran(context,handle,phi,report) bind(C) result(status)
integer(c_int64_t),value :: context,phi
type(c_ptr),value :: handle
type(fort_scope_external_report_v1),intent(out) :: report
integer(c_int) :: status
status=elmm_gpu_pressure_halo_v1(context,handle,phi,report)
end function
function pressure_destroy_from_fortran(handle) bind(C) result(status)
type(c_ptr),intent(inout) :: handle
integer(c_int) :: status
status=elmm_gpu_pressure_destroy_v1(handle)
end function
end module
""")
    run([fc, "-I", str(solver / "modules"), "-I", str(directory), "-c", str(calls),
         "-o", str(directory / "fortran-calls.o")], "fortran-calls")
    references = [native / "P" / (name + ".o") for name in ("customfftw3", "fft", "main", "parameters", "poisfft", "precisions")]
    libraries = [fftw / ("lib" + name + ".so") for name in ("fftw3", "fftw3f", "fftw3_omp", "fftw3f_omp")]
    target = directory / "run"
    run([nvcc, "-ccbin", host, *objects, str(directory / "reference.o"), str(directory / "fortran-calls.o"), *(str(p) for p in references),
         "-L", str(solver / "lib"), "-lpoisson-solver", "-Xlinker=-rpath=" + str(solver / "lib"),
         *(str(p) for p in libraries), "-lgfortran", "-lgomp", "-o", str(target)], "link")
    inputs = [runtime / "scoped_runtime.h", runtime / "scoped_runtime.cu", solver / "lib/libpoisson-solver.so",
              boundary, *references, *libraries]
    receipt = {"metadata": metadata, "commands": commands, "boundary_body_sha256": hashlib.sha256(exact_body.encode()).hexdigest(),
               "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]}
    (directory / "qualification-inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("order", [2, 4])
def test_resident_pressure_chain(pressure_executable, order):
    result = subprocess.run([str(pressure_executable), str(order)], capture_output=True, text=True, timeout=30, check=False,
                            env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    directory = pressure_executable.parent
    (directory / f"fd{order}.stdout").write_text(result.stdout)
    (directory / f"fd{order}.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PRESSURE_DOMAIN_CHAIN_OK FD{order}" in result.stdout
    uploads = re.findall(r"FORT_SCOPED upload buffer=(\d+) generation=17 bytes=(\d+)", result.stderr)
    downloads = re.findall(r"FORT_SCOPED download buffer=(\d+) generation=17 bytes=(\d+)", result.stderr)
    active = (8 * 6 * 10 if order == 2 else 7 * 5 * 9) * 8
    assert sum(int(value) for _, value in uploads) == active
    assert all(identity == "1" for identity, _ in uploads)
    assert not any(identity == "1" for identity, _ in downloads)
    assert result.stderr.count("FORT_SCOPED external_commit ") == 4
    assert result.stderr.count("FORT_SCOPED detach ") == 3


@pytest.mark.cuda
@pytest.mark.parametrize("order", [2, 4])
def test_hidden_device_rejects_before_workspace_and_keeps_original_cpu_operator(pressure_executable, order):
    result = subprocess.run([str(pressure_executable), str(order), "hidden"], capture_output=True, text=True,
                            timeout=30, check=False, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                                                         "FORT_RUNTIME_TRACE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    (pressure_executable.parent / f"fd{order}-hidden.stdout").write_text(result.stdout)
    (pressure_executable.parent / f"fd{order}-hidden.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PRESSURE_HIDDEN_NATIVE_OK FD{order}" in result.stdout
    assert not re.search(r"FORT_SCOPED (upload|download|external_commit|create_cuda|device_allocate) ", result.stderr)
