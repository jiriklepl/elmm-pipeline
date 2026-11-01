#ifndef NONUNIFORM_Z_HPP
#define NONUNIFORM_Z_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::nonuniform_z {
template <std::floating_point T, ValidWorkspace<T> Workspace>
void TridiagZ(Workspace *workspace,
              T *cp_workspace,
              GridSizes<3> ns,
              GridOffsets<2> offs,
              DenomConstPtrs<T, 2> denoms,
              const T *mat_a,
              const T *mat_b,
              const T *mat_c,
              dim3 cu_grid_size,
              dim3 cu_block_size);
}  // namespace poisson_solver::detail::runners::nonuniform_z

#endif  // NONUNIFORM_Z_HPP
