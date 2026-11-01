#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common_cmplx.cuh"
#include "poisson-solver/detail/runners/common_cmplx.hpp"

#include <concepts>

namespace poisson_solver::detail::runners::common_cmplx {
namespace ker = ::poisson_solver::detail::kernels::common_cmplx;

template <std::floating_point T, std::size_t Dim>
void Init(CuComplexTrait<T> *cmplx_workspace,
          const T *real_workspace,
          GridSizes<Dim> ns,
          dim3 cu_grid_size,
          dim3 cu_block_size) {
    ker::Init<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(
        cmplx_workspace, real_workspace, ns);
    CUCH(cudaGetLastError());
}

template <std::floating_point T, std::size_t Dim>
void Normalize(T *res,
               const CuComplexTrait<T> *cmplx_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 cu_grid_size,
               dim3 cu_block_size) {
    ker::Normalize<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(
        res, cmplx_workspace, ns, norm_factor);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_INITIALIZATION(TYPE, DIM)                             \
    template void Init<TYPE, DIM>(CuComplexTrait<TYPE> * cmplx_workspace, \
                                  const TYPE *real_workspace,             \
                                  GridSizes<DIM> ns,                      \
                                  dim3 cu_grid_size,                      \
                                  dim3 cu_block_size);

#define INSTANTIATE_NORMALIZATION(TYPE, DIM)                 \
    template void Normalize<TYPE, DIM>(                      \
        TYPE * res, /* NOLINT(bugprone-macro-parentheses) */ \
        const CuComplexTrait<TYPE> *cmplx_workspace,         \
        GridSizes<DIM> ns,                                   \
        TYPE norm_factor,                                    \
        dim3 cu_grid_size,                                   \
        dim3 cu_block_size);

#define INSTANTIATE_INIT_NORM(TYPE, DIM)  \
    INSTANTIATE_INITIALIZATION(TYPE, DIM) \
    INSTANTIATE_NORMALIZATION(TYPE, DIM)

#define INSTANTIATE_INIT_NORM_ALL_DIMS(TYPE) \
    INSTANTIATE_INIT_NORM(TYPE, 1)           \
    INSTANTIATE_INIT_NORM(TYPE, 2)           \
    INSTANTIATE_INIT_NORM(TYPE, 3)

INSTANTIATE_INIT_NORM_ALL_DIMS(float)
INSTANTIATE_INIT_NORM_ALL_DIMS(double)

#undef INSTANTIATE_INIT_NORM_ALL_DIMS
#undef INSTANTIATE_INIT_NORM
#undef INSTANTIATE_NORMALIZATION
#undef INSTANTIATE_INITIALIZATION
}  // namespace poisson_solver::detail::runners::common_cmplx
