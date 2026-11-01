#ifndef RUNNERS_COMMON_CMPLX_HPP
#define RUNNERS_COMMON_CMPLX_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::common_cmplx {
// Initializes the `cmplx_workspace` with values from `real_workspace`
template <std::floating_point T, std::size_t Dim>
void Init(CuComplexTrait<T> *cmplx_workspace,
          const T *real_workspace,
          GridSizes<Dim> ns,
          dim3 cu_grid_size,
          dim3 cu_block_size);

template <std::floating_point T, std::size_t Dim>
void DivideByDenomsAndZero(CuComplexTrait<T> *cmplx_workspace,
                           GridSizes<Dim> ns,
                           DenomConstPtrs<T, Dim> denoms,
                           dim3 cu_grid_size,
                           dim3 cu_block_size);

template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(CuComplexTrait<T> *cmplx_workspace,
                    GridSizes<Dim> ns,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size);

// Fills `res` with normalized values of `cmplx_workspace`
template <std::floating_point T, std::size_t Dim>
void Normalize(T *res,
               const CuComplexTrait<T> *cmplx_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 cu_grid_size,
               dim3 cu_block_size);
}  // namespace poisson_solver::detail::runners::common_cmplx

#endif  // RUNNERS_COMMON_CMPLX_HPP
