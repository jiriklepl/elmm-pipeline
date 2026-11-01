#ifndef COMMON_CMPLX_CUH
#define COMMON_CMPLX_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::common_cmplx {
template <std::floating_point T>
__device__ auto MakeCuComplex(T r, T i) -> CuComplexTrait<T>;

template <>
__device__ inline auto MakeCuComplex(float r,
                                     float i) -> CuComplexTrait<float> {
    return make_cuFloatComplex(r, i);
}

template <>
__device__ inline auto MakeCuComplex(double r,
                                     double i) -> CuComplexTrait<double> {
    return make_cuDoubleComplex(r, i);
}

template <std::floating_point T, std::size_t Dim>
__global__ void Init(CuComplexTrait<T> *cmplx_workspace,
                     const T *real_workspace,
                     GridSizes<Dim> ns) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        cmplx_workspace[idx] = MakeCuComplex<T>(real_workspace[idx], 0);
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void Normalize(T *res,
                          const CuComplexTrait<T> *cmplx_workspace,
                          GridSizes<Dim> ns,
                          T norm_factor) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        res[idx] = cmplx_workspace[idx].x / norm_factor;
    });
}
}  // namespace poisson_solver::detail::kernels::common_cmplx

#endif  // COMMON_CMPLX_CUH
