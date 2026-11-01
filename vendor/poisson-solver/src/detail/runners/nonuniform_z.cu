#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/nonuniform_z.cuh"
#include "poisson-solver/detail/runners/nonuniform_z.hpp"

#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::nonuniform_z {
namespace nzk = kernels::nonuniform_z;

template <std::floating_point T, ValidWorkspace<T> Workspace>
void TridiagZ(Workspace *real_workspace,
              T *cp_workspace,
              GridSizes<3> ns,
              GridOffsets<2> offs,
              DenomConstPtrs<T, 2> denoms,
              const T *mat_a,
              const T *mat_b,
              const T *mat_c,
              dim3 cu_grid_size,
              dim3 cu_block_size) {
    nzk::TridiagZ<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(
        real_workspace, cp_workspace, ns, offs, denoms, mat_a, mat_b, mat_c);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_TRIDIAG_REAL(TYPE)                                  \
    template void TridiagZ<TYPE>(                                       \
        TYPE * real_workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        TYPE * cp_workspace,   /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<3> ns,                                                \
        GridOffsets<2> offs,                                            \
        DenomConstPtrs<TYPE, 2> denoms,                                 \
        const TYPE *mat_a,                                              \
        const TYPE *mat_b,                                              \
        const TYPE *mat_d,                                              \
        dim3 cu_grid_size,                                              \
        dim3 cu_block_size);

#define INSTANTIATE_TRIDIAG_COMPLEX(TYPE)                             \
    template void TridiagZ<TYPE>(                                     \
        CuComplexTrait<TYPE> * cmplx_workspace,                       \
        TYPE * cp_workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<3> ns,                                              \
        GridOffsets<2> offs,                                          \
        DenomConstPtrs<TYPE, 2> denoms,                               \
        const TYPE *mat_a,                                            \
        const TYPE *mat_b,                                            \
        const TYPE *mat_d,                                            \
        dim3 cu_grid_size,                                            \
        dim3 cu_block_size);

#define INSTANTIATE_TRIDIAG(TYPE)  \
    INSTANTIATE_TRIDIAG_REAL(TYPE) \
    INSTANTIATE_TRIDIAG_COMPLEX(TYPE)

INSTANTIATE_TRIDIAG(float);
INSTANTIATE_TRIDIAG(double);

#undef INSTANTIATE_TRIDIAG
#undef INSTANTIATE_TRIDIAG_COMPLEX
#undef INSTANTIATE_TRIDIAG_REAL
}  // namespace poisson_solver::detail::runners::nonuniform_z
