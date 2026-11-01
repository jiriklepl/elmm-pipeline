#include "gpu_pressure.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

struct elmm_gpu_pressure_handle_v1 {
    fort_scope_domain_t domain=0;
    uint64_t lease=0;
    size_t workspace=0;
    int device=-1;
    poisson_solver_handle_v1 *solver=nullptr;
    elmm_gpu_pressure_config_v1 config{};
    bool ready=false;
    std::mutex mutex;
};
extern "C" int elmm_gpu_pressure_launch_halo_v1(double *,const int32_t *,const int32_t *,void *,int *);
namespace {
thread_local char diagnostic[1024]{};
void message(const char *s) { std::snprintf(diagnostic,sizeof diagnostic,"%s",s?s:""); }
void require(bool condition,const char *s) { if(!condition) throw std::invalid_argument(s); }
void cuda_check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
struct DeviceScope {
    int previous=0;bool changed=false;
    explicit DeviceScope(int ordinal) {
        cuda_check(cudaGetDevice(&previous));
        if(previous!=ordinal) { cuda_check(cudaSetDevice(ordinal));changed=true; }
    }
    void restore() { if(changed) { cuda_check(cudaSetDevice(previous));changed=false; } }
    ~DeviceScope() { if(changed) (void)cudaSetDevice(previous); }
};
int provider_status(int status) {
    if(!status) return FORT_SCOPE_OK;
    message(poisson_solver_last_error_v1());
    if(status==POISSON_SOLVER_ARGUMENT || status==POISSON_SOLVER_UNSUPPORTED) return FORT_SCOPE_ARGUMENT;
    if(status==POISSON_SOLVER_RESOURCE) return FORT_SCOPE_RESOURCE;
    return FORT_SCOPE_EXECUTION;
}
template<class F> int guarded(F action) noexcept {
    diagnostic[0]=0;
    try { return action(); }
    catch(const std::invalid_argument &e) { message(e.what());return FORT_SCOPE_ARGUMENT; }
    catch(const std::bad_alloc &e) { message(e.what());return FORT_SCOPE_RESOURCE; }
    catch(const std::exception &e) { message(e.what());return FORT_SCOPE_EXECUTION; }
    catch(...) { message("unknown pressure adapter error");return FORT_SCOPE_EXECUTION; }
}
void config_check(const elmm_gpu_pressure_config_v1 &c) {
    require(c.version==1 && !c.reserved && c.solver.version==1 && !c.solver.flags,"unknown pressure adapter configuration");
    require(c.solver.dimensions==3 && (c.solver.approximation==2 || c.solver.approximation==4),"pressure adapter requires dimension3 FD2/FD4");
    require(c.solver.device_budget_bytes>0 && c.solver.device_budget_bytes<=SIZE_MAX,"invalid provider workspace cap");
    size_t count=1;
    for(int i=0;i<3;++i) {
        require(c.solver.sizes[i]>=2 && c.solver.sizes[i]<=INT32_MAX-4,"two-layer pressure faces require valid extents >=2");
        require(std::isfinite(c.solver.lengths[i]) && c.solver.lengths[i]>0,"invalid pressure grid length");
        const int a=c.solver.boundary_conditions[2*i],b=c.solver.boundary_conditions[2*i+1];
        require((a==0 || a==4) && a==b,"only periodic/NeumannStag face pairs are qualified");
        const auto extent=size_t(c.solver.sizes[i])+4;
        require(c.phi_lower_bounds[i]<=INT64_MAX-int64_t(extent)+1 &&
                c.rhs_lower_bounds[i]<=INT64_MAX-int64_t(extent-2)+1,"pressure descriptor coordinates overflow");
        require(count<=SIZE_MAX/extent,"pressure layout overflow");count*=extent;
    }
    require(count<=SIZE_MAX/sizeof(double),"pressure layout byte overflow");
}
int retire(elmm_gpu_pressure_handle_v1 &h) {
    // An inactive/failed preflight may never have selected or initialized CUDA.
    std::unique_ptr<DeviceScope> selected;
    if(h.solver) selected=std::make_unique<DeviceScope>(h.device);
    if(h.solver) { const int status=provider_status(poisson_solver_destroy_v1(&h.solver));if(status) return status; }
    if(h.lease) { const int status=fort_scope_domain_workspace_release_v1(h.domain,h.lease);if(status) return status;h.lease=0; }
    if(selected) selected->restore();
    return FORT_SCOPE_OK;
}
void report_reset(fort_scope_external_report_v1 *r) { if(r) *r={1,FORT_SCOPE_ARGUMENT,0,0}; }
int context_check(fort_scope_t context,const elmm_gpu_pressure_handle_v1 &h) {
    fort_scope_domain_t domain=0;const int status=fort_scope_domain_get_v1(context,&domain);
    if(status) return status;
    if(domain!=h.domain || !domain || !h.ready) { message("pressure provider belongs to another domain or is not ready");return FORT_SCOPE_ARGUMENT; }
    return FORT_SCOPE_OK;
}
void view_check(const fort_scope_external_view_v1 &v,const elmm_gpu_pressure_handle_v1 &h,bool phi) {
    require(v.type==FORT_SCOPE_REAL64 && v.rank==3 && v.element_bytes==sizeof(double),"pressure view requires rank3 REAL64");
    require(v.extents && v.strides && v.lower_bounds && v.device,"missing pressure view geometry/storage");
    size_t bytes=sizeof(double);const int ghosts=phi?2:1;
    const auto *lower=phi?h.config.phi_lower_bounds:h.config.rhs_lower_bounds;
    for(int i=0;i<3;++i) {
        require(v.extents[i]==size_t(h.config.solver.sizes[i])+2*ghosts && v.strides[i]==bytes && v.lower_bounds[i]==lower[i],"pressure view differs from checked dense original layout");
        require(bytes<=SIZE_MAX/v.extents[i],"pressure view byte overflow");bytes*=v.extents[i];
    }
    require(v.bytes==bytes,"pressure root bytes differ from full dense layout");
}
int solve_worker(void *opaque,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) noexcept {
    *started=0;
    return guarded([&]() -> int {
        auto &h=*static_cast<elmm_gpu_pressure_handle_v1 *>(opaque);
        require(count==2 && views,"pressure transaction requires Phi and RHS");
        view_check(views[0],h,true);view_check(views[1],h,false);
        const int32_t phi_ghosts[3]={2,2,2},rhs_ghosts[3]={1,1,1};
        const int status=poisson_solver_execute_device_v1(h.solver,views[0].device,views[0].bytes,
            views[1].device,views[1].bytes,phi_ghosts,rhs_ghosts,stream,started);
        return provider_status(status);
    });
}
int halo_worker(void *opaque,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) noexcept {
    *started=0;
    return guarded([&]() -> int {
        auto &h=*static_cast<elmm_gpu_pressure_handle_v1 *>(opaque);
        require(count==1 && views,"pressure halo transaction requires one Phi root");view_check(views[0],h,true);
        return elmm_gpu_pressure_launch_halo_v1(static_cast<double *>(views[0].device),h.config.solver.sizes,
            h.config.solver.boundary_conditions,stream,started);
    });
}
}
extern "C" const char *elmm_gpu_pressure_error_v1(void) { return diagnostic; }
extern "C" int elmm_gpu_pressure_create_v1(fort_scope_domain_t domain,const elmm_gpu_pressure_config_v1 *c,elmm_gpu_pressure_handle_v1 **out) {
    std::unique_ptr<elmm_gpu_pressure_handle_v1> h;
    int status=guarded([&]() -> int {
        require(out,"missing pressure provider output");*out=nullptr;require(c && domain,"missing pressure config/domain");config_check(*c);
        int ordinal=-1;
        int result=fort_scope_domain_device_get_v1(domain,&ordinal);if(result) return result;
        int visible=0;const cudaError_t visibility=cudaGetDeviceCount(&visible);
        if(visibility==cudaErrorNoDevice || (visibility==cudaSuccess && visible==0)) {
            message("pressure device is unavailable before provider creation");return FORT_SCOPE_RESOURCE;
        }
        // Other errors may report earlier asynchronous work failure. They
        // must retain fatal execution semantics rather than permit replay.
        cuda_check(visibility);
        h=std::make_unique<elmm_gpu_pressure_handle_v1>();h->domain=domain;h->config=*c;
        h->device=ordinal;
        result=fort_scope_domain_workspace_reserve_v1(domain,size_t(c->solver.device_budget_bytes),&h->lease);if(result) return result;
        DeviceScope selected(h->device);
        result=provider_status(poisson_solver_create_v1(&c->solver,&h->solver));if(result) { selected.restore();return result; }
        poisson_solver_workspace_v1 bytes{};result=provider_status(poisson_solver_workspace_query_v1(h->solver,&bytes));if(result) { selected.restore();return result; }
        require(bytes.version==1 && bytes.flags==(POISSON_SOLVER_DEVICE_BUFFERS_EXACT|POISSON_SOLVER_DRIVER_OVERHEAD_EXCLUDED) && bytes.execution_extra_bytes==0 && bytes.persistent_device_bytes>0 && bytes.persistent_device_bytes<=bytes.peak_device_bytes && bytes.peak_device_bytes<=c->solver.device_budget_bytes,"unqualified provider allocation accounting");
        require(bytes.persistent_device_bytes<=SIZE_MAX-bytes.execution_extra_bytes,"provider workspace overflow");
        h->workspace=size_t(bytes.persistent_device_bytes+bytes.execution_extra_bytes);
        result=fort_scope_domain_workspace_resize_v1(domain,h->lease,h->workspace);if(result) { selected.restore();return result; }
        selected.restore();h->ready=true;*out=h.release();return FORT_SCOPE_OK;
    });
    if(h) {
        char original[sizeof diagnostic];std::memcpy(original,diagnostic,sizeof original);
        const int cleanup=guarded([&]() -> int { return retire(*h); });
        if(!cleanup) message(original);
        if(cleanup) { h->ready=false;if(out) *out=h.release();return cleanup; }
    }
    return status;
}
extern "C" int elmm_gpu_pressure_solve_v1(fort_scope_t context,elmm_gpu_pressure_handle_v1 *h,fort_buffer_t phi,fort_buffer_t rhs,fort_scope_external_report_v1 *report) {
    report_reset(report);
    const int result=guarded([&]() -> int {
        require(h && report && phi && rhs && phi!=rhs,"missing/aliased pressure roots or provider");std::lock_guard lock(h->mutex);
        const int status=context_check(context,*h);if(status) return status;
        std::array<size_t,3> pl{2,2,2},pu{},rl{1,1,1},ru{};
        for(int i=0;i<3;++i) { pu[i]=size_t(h->config.solver.sizes[i])+2;ru[i]=size_t(h->config.solver.sizes[i])+1; }
        fort_scope_section interior{pl.data(),pu.data()},input{rl.data(),ru.data()};
        fort_scope_access write{0,0,nullptr,1,&interior,1,&interior},read{0,1,&input,0,nullptr,0,nullptr};
        fort_scope_external_binding_v1 bindings[2]={{phi,FORT_SCOPE_REAL64,3,8,write},{rhs,FORT_SCOPE_REAL64,3,8,read}};
        fort_scope_external_call_v1 call{1,0,bindings,2,h->lease,h->workspace};
        return fort_scope_external_execute_v1(context,&call,solve_worker,h,report);
    });
    if(report) report->status=result;return result;
}
extern "C" int elmm_gpu_pressure_halo_v1(fort_scope_t context,elmm_gpu_pressure_handle_v1 *h,fort_buffer_t phi,fort_scope_external_report_v1 *report) {
    report_reset(report);
    const int result=guarded([&]() -> int {
        require(h && report && phi,"missing pressure halo root or provider");std::lock_guard lock(h->mutex);
        const int status=context_check(context,*h);if(status) return status;
        std::array<size_t,3> lower{2,2,2},upper{};for(int i=0;i<3;++i) upper[i]=size_t(h->config.solver.sizes[i])+2;
        std::array<std::array<size_t,3>,6> lo,hi,read_lo,read_hi;
        std::array<fort_scope_section,6> faces,reads;
        for(int axis=0;axis<3;++axis) for(int side=0;side<2;++side) {
            const int f=2*axis+side;lo[f]=lower;hi[f]=upper;
            lo[f][axis]=side?upper[axis]:0;hi[f][axis]=lo[f][axis]+2;faces[f]={lo[f].data(),hi[f].data()};
            read_lo[f]=lower;read_hi[f]=upper;
            const bool periodic=h->config.solver.boundary_conditions[f]==0;
            read_lo[f][axis]=(periodic!=bool(side))?upper[axis]-2:2;
            read_hi[f][axis]=read_lo[f][axis]+2;reads[f]={read_lo[f].data(),read_hi[f].data()};
        }
        fort_scope_access access{0,6,reads.data(),6,faces.data(),6,faces.data()};
        fort_scope_external_binding_v1 binding{phi,FORT_SCOPE_REAL64,3,8,access};
        fort_scope_external_call_v1 call{1,0,&binding,1,0,0};return fort_scope_external_execute_v1(context,&call,halo_worker,h,report);
    });
    if(report) report->status=result;return result;
}
extern "C" int elmm_gpu_pressure_destroy_v1(elmm_gpu_pressure_handle_v1 **out) {
    return guarded([&]() -> int {
        require(out,"missing pressure provider pointer");auto *h=*out;if(!h) return FORT_SCOPE_OK;
        std::unique_lock lock(h->mutex);const int status=retire(*h);if(status) return status;
        lock.unlock();delete h;*out=nullptr;return FORT_SCOPE_OK;
    });
}
