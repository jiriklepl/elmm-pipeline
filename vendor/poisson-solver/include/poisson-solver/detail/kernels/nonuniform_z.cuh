#ifndef NONUNIFORM_Z_CUH
#define NONUNIFORM_Z_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::nonuniform_z {
template <std::floating_point T, ValidWorkspace<T> Workspace>
__global__ void TridiagZ(Workspace *workspace,
                         T *cp_workspace,
                         GridSizes<3> ns,
                         GridOffsets<2> offs,
                         DenomConstPtrs<T, 2> denoms,
                         const T *mat_a,
                         const T *mat_b,
                         const T *mat_c) {
    GridSizes<2> ns_2d = {ns[0], ns[1]};
    ApplyStrideLoop(ns_2d, [&](MultidimIdx<2> midx) {
        auto x = midx[0];
        auto y = midx[1];
        auto idx = FlatIndex(midx, ns_2d);
        auto stride_z = ns[0] * ns[1];
        auto nz = ns[2];

        T c1 = OperatesOnGridOrigin<2>(midx, offs) ? T{0.0} : mat_c[0];
        T lam = denoms[0][x + offs[0]] + denoms[1][y + offs[1]];
        T m = mat_b[0] + lam;

        workspace[idx] /= m;

        if (nz == 1) {
            return;
        }

        cp_workspace[idx] = c1 / m;

        std::size_t prev_idx;
        for (std::size_t z = 1; z < nz - 1; ++z) {
            prev_idx = idx;
            idx += stride_z;

            m = mat_b[z] + lam - cp_workspace[prev_idx] * mat_a[z];
            cp_workspace[idx] = mat_c[z] / m;
            workspace[idx] =
                (workspace[idx] - workspace[prev_idx] * mat_a[z]) / m;
        }
        prev_idx = idx;
        idx += stride_z;
        workspace[idx] =
            (workspace[idx] - workspace[prev_idx] * mat_a[nz - 1]) /
            (mat_b[nz - 1] + lam - cp_workspace[prev_idx] * mat_a[nz - 1]);

        for (std::size_t z = nz - 2; z != static_cast<std::size_t>(-1); --z) {
            std::size_t next_idx = idx;
            idx -= stride_z;
            workspace[idx] -= cp_workspace[idx] * workspace[next_idx];
        }
    });
}
}  // namespace poisson_solver::detail::kernels::nonuniform_z

#endif  // NONUNIFORM_Z_CUH
