#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/kernels/common_real.cuh"
#include "poisson-solver/detail/runners/common_real.hpp"

#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::common_real {
namespace ker = ::poisson_solver::detail::kernels::common_real;

template <std::floating_point T, std::size_t Dim>
void Normalize(T *real_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 cu_grid_size,
               dim3 cu_block_size) {
    ker::Normalize<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(
        real_workspace, ns, norm_factor);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_NORMALIZATION(TYPE, DIM)                            \
    template void Normalize<TYPE, DIM>(                                 \
        TYPE * real_workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<DIM> ns,                                              \
        TYPE norm_factor,                                               \
        dim3 cu_grid_size,                                              \
        dim3 cu_block_size);

#define INSTANTIATE_NORMALIZATION_ALL_DIMS(TYPE) \
    INSTANTIATE_NORMALIZATION(TYPE, 1)           \
    INSTANTIATE_NORMALIZATION(TYPE, 2)           \
    INSTANTIATE_NORMALIZATION(TYPE, 3)

INSTANTIATE_NORMALIZATION_ALL_DIMS(float)
INSTANTIATE_NORMALIZATION_ALL_DIMS(double)

#undef INSTANTIATE_NORMALIZATION_ALL_DIMS
#undef INSTANTIATE_NORMALIZATION
}  // namespace poisson_solver::detail::runners::common_real
