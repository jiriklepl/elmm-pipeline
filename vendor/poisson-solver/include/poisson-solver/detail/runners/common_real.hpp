#ifndef RUNNERS_COMMON_REAL_HPP
#define RUNNERS_COMMON_REAL_HPP

#include "poisson-solver/aliases.hpp"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::common_real {
template <std::floating_point T, std::size_t Dim>
void Normalize(T *real_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 cu_grid_size,
               dim3 cu_block_size);
}  // namespace poisson_solver::detail::runners::common_real

#endif  // RUNNERS_COMMON_REAL_HPP
