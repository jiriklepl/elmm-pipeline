#ifndef COMMON_REAL_CUH
#define COMMON_REAL_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::common_real {
template <std::floating_point T, std::size_t Dim>
__global__ void Normalize(T *real_workspace, GridSizes<Dim> ns, T norm_factor) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        real_workspace[idx] /= norm_factor;
    });
}
}  // namespace poisson_solver::detail::kernels::common_real

#endif  // COMMON_REAL_CUH
