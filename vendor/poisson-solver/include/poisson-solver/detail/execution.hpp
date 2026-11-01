#ifndef POISSON_SOLVER_EXECUTION_HPP
#define POISSON_SOLVER_EXECUTION_HPP

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <unordered_map>

namespace poisson_solver::detail {
// Host thread-local submission state only. Each solver owns its buffers and
// FFT plans; callers must serialize access to one solver instance.
inline thread_local cudaStream_t execution_stream = nullptr;
struct StreamScope {
    cudaStream_t previous;
    explicit StreamScope(cudaStream_t stream) : previous(execution_stream) { execution_stream = stream; }
    ~StreamScope() { execution_stream = previous; }
};
struct AllocationAccount {
    std::unordered_map<void *, std::size_t> live;
    bool budget_exceeded = false;
    bool allocations_forbidden = false, forbidden_allocation_attempted = false;
    std::size_t bytes = 0, peak = 0, limit = std::numeric_limits<std::size_t>::max();
};
inline thread_local AllocationAccount *allocation_account = nullptr;
struct AllocationScope {
    AllocationAccount *previous;
    explicit AllocationScope(AllocationAccount &account) : previous(allocation_account) { allocation_account = &account; }
    ~AllocationScope() { allocation_account = previous; }
};
struct NoAllocationScope {
    AllocationAccount &account;
    bool previous;
    explicit NoAllocationScope(AllocationAccount &value)
        : account(value), previous(value.allocations_forbidden) {
        account.allocations_forbidden = true;
        account.forbidden_allocation_attempted = false;
    }
    ~NoAllocationScope() { account.allocations_forbidden = previous; }
};
inline cudaError_t DeviceAllocate(void **out, std::size_t bytes) noexcept {
    if (allocation_account && allocation_account->allocations_forbidden) {
        allocation_account->forbidden_allocation_attempted = true;
        return cudaErrorMemoryAllocation;
    }
    if (allocation_account && bytes > allocation_account->limit - allocation_account->bytes) {
        allocation_account->budget_exceeded = true;
        return cudaErrorMemoryAllocation;
    }
    const auto status = cudaMalloc(out, bytes);
    if (status != cudaSuccess || !allocation_account) return status;
    try {
        allocation_account->live.emplace(*out, bytes);
        allocation_account->bytes += bytes;
        if (allocation_account->bytes > allocation_account->peak) allocation_account->peak = allocation_account->bytes;
    } catch (...) {
        (void)cudaFree(*out); *out = nullptr; return cudaErrorMemoryAllocation;
    }
    return status;
}
inline cudaError_t DeviceFree(void *ptr) noexcept {
    const auto status = cudaFree(ptr);
    if (status == cudaSuccess && allocation_account) {
        auto found = allocation_account->live.find(ptr);
        if (found != allocation_account->live.end()) {
            allocation_account->bytes -= found->second;
            allocation_account->live.erase(found);
        }
    }
    return status;
}
} // namespace poisson_solver::detail
#endif
