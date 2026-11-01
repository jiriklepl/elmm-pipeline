#include "poisson-solver/detail/execution.hpp"
#include <cstdio>

int main() {
    using namespace poisson_solver::detail;
    AllocationAccount account;
    account.limit = 1;
    AllocationScope allocation(account);
    void *pointer = nullptr;
    {
        NoAllocationScope execute(account);
        if (DeviceAllocate(&pointer, 1) != cudaErrorMemoryAllocation ||
            !account.forbidden_allocation_attempted || pointer || account.bytes || account.peak)
            return 1;
    }
    // Exceeding a known cap must also reject before touching CUDA, so this
    // contract remains executable on a CPU-only packaging host.
    if (account.allocations_forbidden || DeviceAllocate(&pointer, 2) != cudaErrorMemoryAllocation ||
        !account.budget_exceeded || pointer || account.bytes || account.peak)
        return 2;
    std::puts("PASS allocation-free execute and bounded-create preflight");
    return 0;
}
