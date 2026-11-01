"""Independent spatial-stencil qualification of the installed pressure provider."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from elmm_pipeline.gpu_pressure_build import require_pressure_backend

CLIENT = r'''
#include "poisson-solver/device_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void cuda_check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void api(int status) {
    if(status) throw std::runtime_error(poisson_solver_last_error_v1());
}
static void require(bool okay,const char *message) { if(!okay) throw std::runtime_error(message); }
static size_t dense(const std::array<int,3> &n,int i,int j,int k) {
    return i+size_t(n[0])*(j+size_t(n[1])*k);
}
static size_t padded(const std::array<int,3> &n,int g,int i,int j,int k) {
    return i+g+size_t(n[0]+2*g)*(j+g+size_t(n[1]+2*g)*(k+g));
}
static size_t count(const std::array<int,3> &n,int g) {
    return size_t(n[0]+2*g)*(n[1]+2*g)*(n[2]+2*g);
}
// This is a spatial finite-difference oracle. It does not call provider or
// PoisFFT eigenvalue/transform helpers. x/y wrap; z reflects cell centres at
// the two staggered Neumann boundaries (u[-1]=u[0], u[-2]=u[1]).
static int boundary(int i,int n,bool periodic) {
    if(periodic) { i%=n;return i<0?i+n:i; }
    while(i<0 || i>=n) i=i<0?-i-1:2*n-1-i;
    return i;
}
static double sample(const std::vector<double> &u,const std::array<int,3> &n,int i,int j,int k) {
    return u[dense(n,boundary(i,n[0],true),boundary(j,n[1],true),boundary(k,n[2],false))];
}
static std::vector<double> laplacian(const std::vector<double> &u,const std::array<int,3> &n,
                                    const std::array<double,3> &lengths,int order) {
    std::vector<double> result(u.size());
    for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i) {
        double value=0;
        for(int axis=0;axis<3;++axis) {
            int lo[3]={i,j,k},hi[3]={i,j,k},lo2[3]={i,j,k},hi2[3]={i,j,k},lo3[3]={i,j,k},hi3[3]={i,j,k};
            --lo[axis];++hi[axis];lo2[axis]-=2;hi2[axis]+=2;lo3[axis]-=3;hi3[axis]+=3;
            const double center=sample(u,n,i,j,k),left=sample(u,n,lo[0],lo[1],lo[2]);
            const double right=sample(u,n,hi[0],hi[1],hi[2]);
            double second=left-2*center+right;
            // Original PoisFFT FD4 composes staggered fourth-order gradient
            // weights [1,-27,27,-1]/24 with their negative transpose.
            // Its spatial Laplacian has radius3, rather than the usual
            // five-point fourth-order second-derivative approximation.
            if(order==4) second=(sample(u,n,lo3[0],lo3[1],lo3[2])-54*sample(u,n,lo2[0],lo2[1],lo2[2])
                                +783*left-1460*center+783*right-54*sample(u,n,hi2[0],hi2[1],hi2[2])
                                +sample(u,n,hi3[0],hi3[1],hi3[2]))/576;
            const double h=lengths[axis]/n[axis];value+=second/(h*h);
        }
        result[dense(n,i,j,k)]=value;
    }
    return result;
}
static double mean(const std::vector<double> &v) {
    long double total=0;for(double x:v) total+=x;return double(total/v.size());
}
int main(int argc,char **argv) {
  try {
    require(argc==4,"order/grid/kind arguments required");
    const int order=std::atoi(argv[1]),grid=std::atoi(argv[2]),kind=std::atoi(argv[3]);
    require((order==2 || order==4) && (grid==0 || grid==1) && kind>=0 && kind<=2,"bad arguments");
    const std::array<int,3> n=grid?std::array<int,3>{9,7,11}:std::array<int,3>{7,5,9};
    const std::array<double,3> lengths{2.75,1.25,3.5};const double pi=std::acos(-1.0);
    std::vector<double> exact(count(n,0));
    for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i) {
        const double x=2*pi*i/n[0],y=4*pi*j/n[1],z=pi*(k+.5)/n[2];
        exact[dense(n,i,j,k)]=kind==2?0:std::sin(x)+.25*std::cos(y)+.375*std::cos(z)+.2*std::sin(x)*std::cos(z);
    }
    const double gauge=mean(exact);for(auto &x:exact) x-=gauge;
    auto forcing=laplacian(exact,n,lengths,order);
    const double compatibility=mean(forcing);for(auto &x:forcing) x-=compatibility;
    const double offset=kind?0.625:0;for(auto &x:forcing) x+=offset;
    const double rhs_mean=mean(forcing);
    require(std::abs(rhs_mean-offset)<1e-12,"manufactured RHS compatibility differs");
    const double canary=-0x1.23456789abcp+10;
    const size_t pcount=count(n,2),rcount=count(n,1);
    std::vector<double> phi(pcount+2,canary),rhs(rcount+2,canary);
    for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i)
        rhs[1+padded(n,1,i,j,k)]=forcing[dense(n,i,j,k)];
    const auto original_rhs=rhs;
    poisson_solver_config_v1 config{};config.version=1;config.dimensions=3;config.approximation=order;
    config.device_budget_bytes=4ULL<<20;std::copy(n.begin(),n.end(),config.sizes);
    std::copy(lengths.begin(),lengths.end(),config.lengths);config.boundary_conditions[4]=config.boundary_conditions[5]=4;
    poisson_solver_handle_v1 *solver=nullptr;api(poisson_solver_create_v1(&config,&solver));
    poisson_solver_workspace_v1 before{};api(poisson_solver_workspace_query_v1(solver,&before));
    cudaStream_t stream;cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    double *device_phi=nullptr,*device_rhs=nullptr;
    cuda_check(cudaMalloc(&device_phi,phi.size()*8));cuda_check(cudaMalloc(&device_rhs,rhs.size()*8));
    cuda_check(cudaMemcpyAsync(device_phi,phi.data(),phi.size()*8,cudaMemcpyHostToDevice,stream));
    cuda_check(cudaMemcpyAsync(device_rhs,rhs.data(),rhs.size()*8,cudaMemcpyHostToDevice,stream));
    const int32_t gp[3]={2,2,2},gr[3]={1,1,1};int started=-1;
    api(poisson_solver_execute_device_v1(solver,device_phi+1,pcount*8,device_rhs+1,rcount*8,gp,gr,stream,&started));
    require(started==1,"solve missing submitted marker");
    cuda_check(cudaMemcpyAsync(phi.data(),device_phi,phi.size()*8,cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaMemcpyAsync(rhs.data(),device_rhs,rhs.size()*8,cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaStreamSynchronize(stream));
    require(!std::memcmp(rhs.data(),original_rhs.data(),rhs.size()*8),"RHS/ghost/canary bytes changed");
    require(!std::memcmp(phi.data(),&canary,8) && !std::memcmp(phi.data()+phi.size()-1,&canary,8),"Phi allocation canary changed");
    std::vector<double> actual(exact.size());double error=0;
    for(int k=-2;k<n[2]+2;++k) for(int j=-2;j<n[1]+2;++j) for(int i=-2;i<n[0]+2;++i) {
        const size_t q=1+padded(n,2,i,j,k);
        if(i<0 || i>=n[0] || j<0 || j>=n[1] || k<0 || k>=n[2]) {
            require(!std::memcmp(phi.data()+q,&canary,8),"Phi halo bytes changed");continue;
        }
        require(std::isfinite(phi[q]),"nonfinite manufactured solution");actual[dense(n,i,j,k)]=phi[q];
        error=std::max(error,std::abs(phi[q]-exact[dense(n,i,j,k)]));
    }
    const auto discrete=laplacian(actual,n,lengths,order),wrong=laplacian(actual,n,lengths,order==4?2:4);
    double residual=0,wrong_residual=0,scale=1;
    for(size_t q=0;q<forcing.size();++q) {
        const double projected=forcing[q]-rhs_mean;
        residual=std::max(residual,std::abs(discrete[q]-projected));
        wrong_residual=std::max(wrong_residual,std::abs(wrong[q]-projected));scale=std::max(scale,std::abs(projected));
    }
    require(error<2e-10,"manufactured solution error");
    require(residual<5e-10*scale,"independent discrete stencil residual");
    require(std::abs(mean(actual))<2e-10,"nonzero nullspace gauge");
    if(kind!=2) require(wrong_residual>1e-4*scale,"FD2/FD4 oracle not discriminating");
    poisson_solver_workspace_v1 after{};api(poisson_solver_workspace_query_v1(solver,&after));
    require(before.persistent_device_bytes==after.persistent_device_bytes && before.peak_device_bytes==after.peak_device_bytes
            && !after.execution_extra_bytes,"unexpected execution workspace allocation");
    api(poisson_solver_destroy_v1(&solver));require(!solver,"solver handle retained");
    cuda_check(cudaFree(device_phi));cuda_check(cudaFree(device_rhs));cuda_check(cudaStreamDestroy(stream));
    std::printf("{\"order\":%d,\"grid\":[%d,%d,%d],\"kind\":%d,\"max_error\":%.17g,\"residual\":%.17g,"
                "\"residual_scale\":%.17g,\"wrong_stencil_residual\":%.17g,\"gauge\":%.17g,\"rhs_mean\":%.17g,"
                "\"rhs_compatibility_roundoff\":%.17g,\"rhs_ghost_canaries_unchanged\":true,\"phi_halo_canaries_unchanged\":true,"
                "\"nondefault_stream\":true}\n",order,n[0],n[1],n[2],kind,error,residual,scale,wrong_residual,mean(actual),rhs_mean,compatibility);
  } catch(const std::exception &error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
'''


@pytest.fixture(scope="module")
def manufactured_executable(tmp_path_factory):
    manifest, digest = (os.environ.get(key) for key in ("ELMM_PRESSURE_BACKEND_MANIFEST", "ELMM_PRESSURE_BACKEND_SHA256"))
    if not manifest or not digest:
        pytest.skip("Explicit authenticated pressure backend required")
    fc, nvcc, host = (shutil.which(name) for name in ("gfortran-15", "nvcc", "g++-14"))
    assert fc and nvcc and host
    backend = require_pressure_backend({"manifest": manifest, "sha256": digest}, fortran=fc)
    library = Path(backend["provider_library"]["path"])
    assert (library.parent / "libpoisson-solver.so").resolve(strict=True) == library.resolve(strict=True)
    include = library.parent.parent / "include"
    source = include / "poisson-solver/device_api.h"
    assert source.is_file()
    directory = tmp_path_factory.mktemp("manufactured_pressure")
    client, binary = directory / "client.cu", directory / "run"
    client.write_text(CLIENT)
    argv = [nvcc, "-ccbin", host, "-std=c++17", "-arch=sm_86", "--fmad=false", "-I", str(include),
            str(client), "-L", str(library.parent), "-lpoisson-solver",
            "-Xlinker=-rpath=" + str(library.parent), "-o", str(binary)]
    result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=120, check=False)
    (directory / "compile.stdout").write_text(result.stdout)
    (directory / "compile.stderr").write_text(result.stderr)
    (directory / "compile.json").write_text(json.dumps({"argv": argv, "returncode": result.returncode,
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in
                   (Path(manifest), library, source, client)]}, indent=2) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.cuda
@pytest.mark.parametrize(("grid", "kind"), [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
@pytest.mark.parametrize("order", [2, 4])
def test_manufactured_solution_and_independent_discrete_residual(manufactured_executable, order, grid, kind):
    result = subprocess.run([str(manufactured_executable), str(order), str(grid), str(kind)],
                            capture_output=True, text=True, timeout=30, check=False)
    stem = manufactured_executable.parent / f"fd{order}-grid{grid}-kind{kind}"
    stem.with_suffix(".stdout").write_text(result.stdout)
    stem.with_suffix(".stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    observation = json.loads(result.stdout)
    assert observation["order"] == order
    assert observation["grid"] == ([7, 5, 9] if grid == 0 else [9, 7, 11])
    assert observation["kind"] == kind
    assert observation["rhs_ghost_canaries_unchanged"] is True
    assert observation["phi_halo_canaries_unchanged"] is True
    assert observation["nondefault_stream"] is True
