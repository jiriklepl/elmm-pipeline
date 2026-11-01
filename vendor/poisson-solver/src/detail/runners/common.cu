#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"
#include "poisson-solver/detail/runners/common.hpp"

#include <concepts>

namespace poisson_solver::detail::runners {
namespace ker = ::poisson_solver::detail::kernels;

template <std::floating_point T, std::size_t Dim, ValidWorkspace<T> Workspace>
void DivideByDenoms(Workspace *workspace,
                    GridSizes<Dim> ns,
                    GridOffsets<Dim> offs,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size,
                    bool zero_origin) {
    ker::DivideByDenoms<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(
        workspace, ns, offs, denoms, zero_origin);
    CUCH(cudaGetLastError());
}

template <std::floating_point T, std::size_t Dim>
void UnpackPaddedGrid(T *active, const T * padded, GridSizes<Dim> ns, GridSizes<Dim> ngs, dim3 cu_grid_size, dim3 cu_block_size) {
    ker::UnpackPaddedGrid<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(active, padded, ns, ngs);
    CUCH(cudaGetLastError());
}

template <std::floating_point T, std::size_t Dim>
void PackPaddedGrid(T * padded, const T * active, GridSizes<Dim> ns, GridSizes<Dim> ngs, dim3 cu_grid_size, dim3 cu_block_size) {
    ker::PackPaddedGrid<<<cu_grid_size, cu_block_size, 0, detail::execution_stream>>>(padded, active, ns, ngs);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_DIVISION_REAL(TYPE, DIM)                       \
    template void DivideByDenoms<TYPE, DIM, TYPE>(                 \
        TYPE * workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<DIM> ns,                                         \
        GridOffsets<DIM> offs,                                     \
        DenomConstPtrs<TYPE, DIM> denoms,                          \
        dim3 cu_grid_size,                                         \
        dim3 cu_block_size,                                        \
        bool zero_origin);

#define INSTANTIATE_DIVISION_COMPLEX(TYPE, DIM)                                \
    template void                                                              \
    DivideByDenoms<TYPE,                                                       \
                   DIM,                                                        \
                   CuComplexTrait<TYPE>>(/* NOLINT(bugprone-macro-parentheses) \
                                          */                                   \
                                         CuComplexTrait<TYPE> * workspace,     \
                                         GridSizes<DIM> ns,                    \
                                         GridOffsets<DIM> offs,                \
                                         DenomConstPtrs<TYPE, DIM> denoms,     \
                                         dim3 cu_grid_size,                    \
                                         dim3 cu_block_size,                   \
                                         bool zero_origin);
#define INSTANTIATE_PACKING(TYPE, DIM) \
    template void UnpackPaddedGrid<TYPE, DIM>(TYPE *active, \
            const TYPE *padded, \
            GridSizes<DIM> ns, \
            GridSizes<DIM> ngs, \
            dim3 cu_grid_size, \
            dim3 cu_block_size); \
    template void PackPaddedGrid<TYPE, DIM>(TYPE *padded, \
            const TYPE *active, \
            GridSizes<DIM> ns, \
            GridSizes<DIM> ngs, \
            dim3 cu_grid_size, \
            dim3 cu_block_size);

#define INSTANTIATE_DIV_REAL_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_REAL(TYPE, 1)      \
    INSTANTIATE_DIVISION_REAL(TYPE, 2)      \
    INSTANTIATE_DIVISION_REAL(TYPE, 3)

#define INSTANTIATE_DIV_COMPLEX_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_COMPLEX(TYPE, 1)      \
    INSTANTIATE_DIVISION_COMPLEX(TYPE, 2)      \
    INSTANTIATE_DIVISION_COMPLEX(TYPE, 3)

#define INSTANTIATE_PACKING_ALL_DIMS(TYPE) \
    INSTANTIATE_PACKING(TYPE, 1) \
    INSTANTIATE_PACKING(TYPE, 2) \
    INSTANTIATE_PACKING(TYPE, 3)

INSTANTIATE_DIV_REAL_ALL_DIMS(float)
INSTANTIATE_DIV_REAL_ALL_DIMS(double)
INSTANTIATE_DIV_COMPLEX_ALL_DIMS(float)
INSTANTIATE_DIV_COMPLEX_ALL_DIMS(double)

INSTANTIATE_PACKING_ALL_DIMS(float)
INSTANTIATE_PACKING_ALL_DIMS(double)

#undef INSTANTIATE_PACKING_ALL_DIMS
#undef INSTANTIATE_DIV_COMPLEX_ALL_DIMS
#undef INSTANTIATE_DIV_REAL_ALL_DIMS
#undef INSTANTIATE_PACKING
#undef INSTANTIATE_DIVISION_COMPLEX
#undef INSTANTIATE_DIVISION_REAL
}  // namespace poisson_solver::detail::runners
