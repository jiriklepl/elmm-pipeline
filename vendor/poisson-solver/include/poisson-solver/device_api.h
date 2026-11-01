#ifndef POISSON_SOLVER_DEVICE_API_H
#define POISSON_SOLVER_DEVICE_API_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct poisson_solver_handle_v1 poisson_solver_handle_v1;
enum poisson_solver_status_v1 {
    POISSON_SOLVER_OK = 0, POISSON_SOLVER_ARGUMENT = 1,
    POISSON_SOLVER_RESOURCE = 2, POISSON_SOLVER_EXECUTION = 3,
    POISSON_SOLVER_UNSUPPORTED = 4, POISSON_SOLVER_POISONED = 5
};
typedef struct poisson_solver_config_v1 {
    uint32_t version, dimensions;
    int32_t sizes[3];
    double lengths[3];
    int32_t boundary_conditions[6]; /* same 0..4 enum as c_binding.h */
    int32_t approximation; /* FD2=2 or FD4=4 */
    uint32_t flags; /* must be zero */
    uint64_t device_budget_bytes; /* hard cap for all owned device buffers */
} poisson_solver_config_v1;
typedef struct poisson_solver_workspace_v1 {
    uint32_t version, flags;
    uint64_t persistent_device_bytes, peak_device_bytes, execution_extra_bytes;
    /* Current live buffers; observed creation/lifetime high-water; additional
     * execution requirement (zero for v1), respectively. */
} poisson_solver_workspace_v1;
enum { POISSON_SOLVER_DEVICE_BUFFERS_EXACT = 1, POISSON_SOLVER_DRIVER_OVERHEAD_EXCLUDED = 2 };
/* Local uniform double precision, dimension3. No exceptions cross this ABI.
 * Reserve device_budget_bytes in the caller's residency budget BEFORE create,
 * then retain persistent_device_bytes + execution_extra_bytes. The observed
 * creation peak is diagnostic, not a permanently resident byte claim. CUDA
 * context, module/code and driver bookkeeping memory is outside this ledger. */
int poisson_solver_create_v1(const poisson_solver_config_v1 *, poisson_solver_handle_v1 **);
int poisson_solver_workspace_query_v1(poisson_solver_handle_v1 *, poisson_solver_workspace_v1 *);
/* Dense first-index-fastest padded device allocations. Writes ONLY the active
 * phi interior; preserves every RHS byte and phi halo. Stream is cudaStream_t
 * transported as void*. Success is asynchronous on that stream. started is
 * set0 on entry and1 BEFORE the first enqueue attempt; started failure poisons
 * the handle and forbids native replay. This solver instance serializes calls
 * and orders workspace reuse across streams using a private completion event.
 * Device and allocation lifetime must outlive completion. */
int poisson_solver_execute_device_v1(poisson_solver_handle_v1 *, void *phi, size_t phi_bytes,
    const void *rhs, size_t rhs_bytes, const int32_t ghost_phi[3], const int32_t ghost_rhs[3],
    void *stream, int *started);
/* Waits only this provider's last completion before freeing its workspaces. */
int poisson_solver_destroy_v1(poisson_solver_handle_v1 **);
/* Thread-local diagnostic, valid until the next API call on this thread. */
const char *poisson_solver_last_error_v1(void);
#ifdef __cplusplus
}
#endif
#endif
