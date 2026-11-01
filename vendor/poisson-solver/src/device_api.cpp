#include "poisson-solver/device_api.h"
#include "poisson-solver/solver.hpp"
#include <climits>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>

struct poisson_solver_handle_v1 {
    poisson_solver::detail::AllocationAccount account;
    std::unique_ptr<poisson_solver::Solver<double, 3>> solver;
    poisson_solver::GridSizes<3> sizes;
    cudaEvent_t complete = nullptr;
    bool pending = false, poisoned = false, untracked_submission = false;
    cudaStream_t last_stream = nullptr;
    int device = 0;
    std::mutex mutex;
};
namespace {
thread_local char last_error[1024]{};
void require(bool value, const char *message) { if (!value) throw std::invalid_argument(message); }
void cuda_check(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
template<class F> int protect(F action) noexcept {
    last_error[0] = 0;
    try { return action(); }
    catch (const std::invalid_argument &e) { std::snprintf(last_error,sizeof last_error,"%s",e.what()); return POISSON_SOLVER_ARGUMENT; }
    catch (const std::bad_alloc &e) { std::snprintf(last_error,sizeof last_error,"%s",e.what()); return POISSON_SOLVER_RESOURCE; }
    catch (const std::exception &e) { std::snprintf(last_error,sizeof last_error,"%s",e.what()); return POISSON_SOLVER_EXECUTION; }
    catch (...) { std::snprintf(last_error,sizeof last_error,"unknown solver failure"); return POISSON_SOLVER_EXECUTION; }
}
size_t padded_count(const poisson_solver_handle_v1 &h, const int32_t *ghosts) {
    require(ghosts, "missing ghost counts"); size_t count=1;
    for (size_t i=0;i<3;++i) {
        require(ghosts[i]>=0 && size_t(ghosts[i])<=(INT_MAX-h.sizes[i])/2,"invalid/overflowing ghost extent");
        const auto extent=h.sizes[i]+2*size_t(ghosts[i]);
        require(count<=SIZE_MAX/extent,"padded grid size overflow"); count*=extent;
    }
    require(count<=SIZE_MAX/sizeof(double),"padded grid byte overflow"); return count;
}
void device_pointer(const void *p, int device) {
    require(p,"missing device pointer"); cudaPointerAttributes a{}; cuda_check(cudaPointerGetAttributes(&a,p));
    require(a.type==cudaMemoryTypeDevice && a.device==device,"pointer is not on solver device");
}
}
extern "C" const char *poisson_solver_last_error_v1() { return last_error; }
extern "C" int poisson_solver_create_v1(const poisson_solver_config_v1 *c, poisson_solver_handle_v1 **out) {
    return protect([&] {
        require(out,"missing solver output"); *out=nullptr;
        require(c && c->version==1 && c->flags==0,"unsupported solver configuration");
        if (c->dimensions!=3 || (c->approximation!=2 && c->approximation!=4)) return int(POISSON_SOLVER_UNSUPPORTED);
#ifdef POISSON_SOLVER_USE_MPI
        return int(POISSON_SOLVER_UNSUPPORTED);
#else
        require(c->device_budget_bytes && c->device_budget_bytes<=SIZE_MAX,"invalid provider device budget");
        auto h=std::make_unique<poisson_solver_handle_v1>();
        h->account.limit=size_t(c->device_budget_bytes); size_t count=1;
        std::array<double,3> lengths{}; poisson_solver::BCs<3> bcs{};
        for (size_t i=0;i<3;++i) {
            require(c->sizes[i]>0 && std::isfinite(c->lengths[i]) && c->lengths[i]>0,"invalid grid size or length");
            h->sizes[i]=size_t(c->sizes[i]); lengths[i]=c->lengths[i];
            require(count<=SIZE_MAX/h->sizes[i],"active grid overflow"); count*=h->sizes[i];
            int a=c->boundary_conditions[2*i],b=c->boundary_conditions[2*i+1];
            require(a>=0 && a<=4 && b>=0 && b<=4 && ((a==0)==(b==0)),"invalid boundary pair");
            bcs[i]={poisson_solver::BC(a),poisson_solver::BC(b)};
        }
        require(count<=SIZE_MAX/(4*sizeof(double)),"workspace size overflow");
        if (count > h->account.limit / sizeof(double)) {
            std::snprintf(last_error,sizeof last_error,"provider budget smaller than active workspace");
            return int(POISSON_SOLVER_RESOURCE);
        }
        cuda_check(cudaGetDevice(&h->device));
        poisson_solver::detail::AllocationScope allocation(h->account);
        try {
            if (c->approximation==2) h->solver=std::make_unique<poisson_solver::Solver<double,3>>(h->sizes,lengths,bcs,poisson_solver::FiniteDifference2Approximation<double>{});
            else h->solver=std::make_unique<poisson_solver::Solver<double,3>>(h->sizes,lengths,bcs,poisson_solver::FiniteDifference4Approximation<double>{});
            cuda_check(cudaEventCreateWithFlags(&h->complete,cudaEventDisableTiming));
        } catch (...) { h->solver.reset(); if (h->account.budget_exceeded) throw std::bad_alloc(); throw; }
        *out=h.release(); return int(POISSON_SOLVER_OK);
#endif
    });
}
extern "C" int poisson_solver_workspace_query_v1(poisson_solver_handle_v1 *h, poisson_solver_workspace_v1 *out) {
    return protect([&] {
        require(h && out,"missing solver/workspace output"); std::lock_guard lock(h->mutex);
        *out={1,POISSON_SOLVER_DEVICE_BUFFERS_EXACT|POISSON_SOLVER_DRIVER_OVERHEAD_EXCLUDED,
            h->account.bytes,h->account.peak,0}; return int(POISSON_SOLVER_OK);
    });
}
extern "C" int poisson_solver_execute_device_v1(poisson_solver_handle_v1 *h, void *phi,size_t phi_bytes,
    const void *rhs,size_t rhs_bytes,const int32_t *gp,const int32_t *gr,void *raw_stream,int *started) {
    if (started) *started=0;
    return protect([&] {
        require(h && started,"missing solver/started output"); std::lock_guard lock(h->mutex);
        if (h->poisoned) return int(POISSON_SOLVER_POISONED);
        int device; cuda_check(cudaGetDevice(&device)); require(device==h->device,"solver device changed");
        size_t np=padded_count(*h,gp),nr=padded_count(*h,gr);
        require(phi_bytes==np*sizeof(double) && rhs_bytes==nr*sizeof(double),"padded allocation byte count mismatch");
        const auto p=reinterpret_cast<uintptr_t>(phi),r=reinterpret_cast<uintptr_t>(rhs);
        require(p<=UINTPTR_MAX-phi_bytes && r<=UINTPTR_MAX-rhs_bytes && (p+phi_bytes<=r || r+rhs_bytes<=p),"overlapping solver inputs/output");
        device_pointer(phi,h->device); device_pointer(rhs,h->device);
        auto stream=reinterpret_cast<cudaStream_t>(raw_stream);
        poisson_solver::detail::AllocationScope allocation(h->account);
        const auto resident=h->account.bytes;
        poisson_solver::detail::NoAllocationScope no_allocation(h->account);
        try {
            h->last_stream=stream; h->untracked_submission=true;
            *started=1;
            if (h->pending) cuda_check(cudaStreamWaitEvent(stream,h->complete,0));
            h->solver->ExecuteGPU(std::span(static_cast<double*>(phi),np),std::span(static_cast<const double*>(rhs),nr),
                {size_t(gp[0]),size_t(gp[1]),size_t(gp[2])},{size_t(gr[0]),size_t(gr[1]),size_t(gr[2])},stream);
            cuda_check(cudaEventRecord(h->complete,stream)); h->pending=true; h->untracked_submission=false;
            if (h->account.forbidden_allocation_attempted) throw std::runtime_error("execution attempted unreserved workspace allocation");
            if (h->account.bytes!=resident) throw std::runtime_error("unexpected execution workspace growth");
        } catch (...) { h->poisoned=true; throw; }
        return int(POISSON_SOLVER_OK);
    });
}
extern "C" int poisson_solver_destroy_v1(poisson_solver_handle_v1 **out) {
    return protect([&] {
        require(out,"missing solver pointer"); auto *h=*out; if (!h) return int(POISSON_SOLVER_OK);
        std::unique_lock lock(h->mutex); int device; cuda_check(cudaGetDevice(&device)); require(device==h->device,"solver device changed");
        // A failed enqueue/event-record attempt may have work not represented by
        // complete. Do not release workspaces until its supplied stream joins.
        if (h->untracked_submission) cuda_check(cudaStreamSynchronize(h->last_stream));
        else if (h->pending) cuda_check(cudaEventSynchronize(h->complete));
        poisson_solver::detail::AllocationScope allocation(h->account); h->solver.reset();
        // CUDA destructors cannot throw. Retain the account/handle on a failed
        // free so retirement can report the error and be retried safely.
        while (!h->account.live.empty())
            cuda_check(poisson_solver::detail::DeviceFree(h->account.live.begin()->first));
        if (h->account.bytes) throw std::runtime_error("provider device allocation retirement failed");
        if (h->complete) { cuda_check(cudaEventDestroy(h->complete)); h->complete=nullptr; }
        lock.unlock(); delete h; *out=nullptr; return int(POISSON_SOLVER_OK);
    });
}
