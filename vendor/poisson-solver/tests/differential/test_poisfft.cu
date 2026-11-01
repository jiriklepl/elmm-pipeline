#include "poisson-solver/device_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void reference_solve(const int*,const double*,const int*,int,void*,void*);
static void check(cudaError_t x) { if (x!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void api(int x) { if(x) throw std::runtime_error(std::to_string(x)+": "+poisson_solver_last_error_v1()); }
static size_t count(std::array<int,3> n,int g) { return size_t(n[0]+2*g)*(n[1]+2*g)*(n[2]+2*g); }
static size_t index(std::array<int,3> n,int g,int i,int j,int k) { return (i+g)+size_t(n[0]+2*g)*((j+g)+size_t(n[1]+2*g)*(k+g)); }
// A nonblocking stream's producer must precede ALL solver work. A wrong
// default-stream unpack can observe old storage even though the host call order
// is correct. This is an ordering correctness check, not a timing assertion.
__global__ void delayed_rhs(double *out,const double *in,size_t n) {
    auto begin=clock64(); while(clock64()-begin<20000000ULL) {}
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x) out[i]=in[i];
}
static void one(std::array<int,3> n,std::array<int,6> bcs,int approximation,bool null_rhs,bool cross_stream=false) {
    const double sentinel=-0x1.23456789abcp+10;
    std::array<double,3> lens{2.75,1.25,3.5};
    std::vector<double> rhs(count(n,1),sentinel),initial(count(n,2),sentinel);
    double mean=0;
    for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i) {
        double value=null_rhs ? 1.0 : std::sin(.31*i+.17*j)+std::cos(.27*k+.1*i)+.013*i*j;
        rhs[index(n,1,i,j,k)]=value;mean+=value;
    }
    mean/=double(n[0]*n[1]*n[2]);
    if(!null_rhs) for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i) rhs[index(n,1,i,j,k)]-=mean;
    const auto original_rhs=rhs;auto expected=initial;
    reference_solve(n.data(),lens.data(),bcs.data(),approximation,expected.data(),rhs.data());
    if(rhs!=original_rhs) throw std::runtime_error("PoisFFT reference mutated RHS");
    poisson_solver_config_v1 config{};config.version=1;config.dimensions=3;config.approximation=approximation;config.device_budget_bytes=256ULL<<20;
    std::copy(n.begin(),n.end(),config.sizes);std::copy(lens.begin(),lens.end(),config.lengths);std::copy(bcs.begin(),bcs.end(),config.boundary_conditions);
    poisson_solver_handle_v1 *solver=nullptr;api(poisson_solver_create_v1(&config,&solver));
    poisson_solver_workspace_v1 before{};api(poisson_solver_workspace_query_v1(solver,&before));
    if(!before.persistent_device_bytes || before.peak_device_bytes<before.persistent_device_bytes || before.execution_extra_bytes) throw std::runtime_error("incomplete workspace accounting");
    cudaStream_t stream,second=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    double *p=nullptr,*r=nullptr,*producer=nullptr,*p2=nullptr,*r2=nullptr;
    auto rhs2=original_rhs;
    if(cross_stream) {
        check(cudaStreamCreateWithFlags(&second,cudaStreamNonBlocking));
        check(cudaMalloc(&p2,initial.size()*8));check(cudaMalloc(&r2,rhs.size()*8));
        for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i) rhs2[index(n,1,i,j,k)]*=.5;
        check(cudaMemcpyAsync(p2,initial.data(),initial.size()*8,cudaMemcpyHostToDevice,second));
        check(cudaMemcpyAsync(r2,rhs2.data(),rhs.size()*8,cudaMemcpyHostToDevice,second));
    }
    check(cudaMalloc(&p,initial.size()*8));check(cudaMalloc(&r,rhs.size()*8));check(cudaMalloc(&producer,rhs.size()*8));
    check(cudaMemcpyAsync(p,initial.data(),initial.size()*8,cudaMemcpyHostToDevice,stream));
    check(cudaMemsetAsync(r,0,rhs.size()*8,stream));
    check(cudaMemcpyAsync(producer,original_rhs.data(),rhs.size()*8,cudaMemcpyHostToDevice,stream));
    delayed_rhs<<<1,256,0,stream>>>(r,producer,rhs.size());check(cudaGetLastError());
    int32_t gp[3]={2,2,2},gr[3]={1,1,1};int started=-1;
    api(poisson_solver_execute_device_v1(solver,p,initial.size()*8,r,rhs.size()*8,gp,gr,stream,&started));
    if(started!=1) throw std::runtime_error("successful solve did not report enqueue");
    if(cross_stream) {
        started=-1;
        api(poisson_solver_execute_device_v1(solver,p2,initial.size()*8,r2,rhs.size()*8,gp,gr,second,&started));
        if(started!=1) throw std::runtime_error("cross-stream solve did not report enqueue");
    }
    std::vector<double> actual(initial.size()),after_rhs(rhs.size());
    check(cudaMemcpyAsync(actual.data(),p,actual.size()*8,cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(after_rhs.data(),r,after_rhs.size()*8,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    if(std::memcmp(after_rhs.data(),original_rhs.data(),rhs.size()*8)) throw std::runtime_error("device RHS or halo mutated");
    auto compare=[&](const std::vector<double> &values,double scale) {
    double max_error=0,solution_mean=0;
    for(int k=-2;k<n[2]+2;++k) for(int j=-2;j<n[1]+2;++j) for(int i=-2;i<n[0]+2;++i) {
        auto a=index(n,2,i,j,k);bool inside=i>=0&&i<n[0]&&j>=0&&j<n[1]&&k>=0&&k<n[2];
        if(!inside) { if(std::bit_cast<uint64_t>(values[a])!=std::bit_cast<uint64_t>(sentinel)) throw std::runtime_error("phi halo overwritten");continue; }
        if(!std::isfinite(values[a]) || !std::isfinite(expected[a])) throw std::runtime_error("nonfinite solution");
        double error=std::abs(values[a]-scale*expected[a]);max_error=std::max(max_error,error);solution_mean+=values[a];
        if(error>2e-10+2e-10*std::abs(expected[a])) throw std::runtime_error("FD operator differs from pinned PoisFFT");
    }
    solution_mean/=double(n[0]*n[1]*n[2]);if(std::abs(solution_mean)>2e-10) throw std::runtime_error("nullspace gauge is not zero mean");
    return std::array<double,2>{max_error,solution_mean};
    };
    const auto result=compare(actual,1.0);
    if(cross_stream) {
        check(cudaMemcpyAsync(actual.data(),p2,actual.size()*8,cudaMemcpyDeviceToHost,second));
        check(cudaMemcpyAsync(after_rhs.data(),r2,after_rhs.size()*8,cudaMemcpyDeviceToHost,second));check(cudaStreamSynchronize(second));
        if(std::memcmp(after_rhs.data(),rhs2.data(),rhs2.size()*8)) throw std::runtime_error("second-stream RHS or halo mutated");
        (void)compare(actual,.5);
    }
    poisson_solver_workspace_v1 after{};api(poisson_solver_workspace_query_v1(solver,&after));
    if(after.persistent_device_bytes!=before.persistent_device_bytes || after.peak_device_bytes!=before.peak_device_bytes) throw std::runtime_error("execution allocated unreported workspace");
    api(poisson_solver_destroy_v1(&solver));if(solver) throw std::runtime_error("destroy retained handle");
    check(cudaFree(p));check(cudaFree(r));check(cudaFree(producer));check(cudaStreamDestroy(stream));
    if(cross_stream) { check(cudaFree(p2));check(cudaFree(r2));check(cudaStreamDestroy(second)); }
    std::printf("PASS FD%d n=%d,%d,%d BC=%d%d%d%d%d%d null=%d cross_stream=%d max_error=%.17g mean=%.17g workspace=%llu peak=%llu\n",approximation,n[0],n[1],n[2],bcs[0],bcs[1],bcs[2],bcs[3],bcs[4],bcs[5],null_rhs,cross_stream,result[0],result[1],(unsigned long long)before.persistent_device_bytes,(unsigned long long)before.peak_device_bytes);
}
int main() {
    try {
        for(int order:{2,4}) for(auto n:{std::array<int,3>{8,6,10},std::array<int,3>{7,5,9}})
            for(auto bcs:{std::array<int,6>{0,0,0,0,4,4},std::array<int,6>{0,0,0,0,0,0},std::array<int,6>{4,4,4,4,4,4}}) one(n,bcs,order,false,order==2 && n[0]==8 && bcs[4]==4 && bcs[0]==0);
        for(int order:{2,4}) one({7,5,9},{0,0,0,0,4,4},order,true);
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
    return 0;
}
