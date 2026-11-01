#ifndef COMMON_CUH
#define COMMON_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <utility>
#include <type_traits>

namespace poisson_solver::detail::kernels {
// === Concepts ===
// Needed for operators without `std::floating_point T` parameter.
// `CuComplexTrait<T>` would not be deducible in those.
template <typename C>
concept CuComplexType =
    std::is_same_v<C, cuComplex> || std::is_same_v<C, cuDoubleComplex>;

// === cu(Double)Complex Operators ===
// These facilitate kernel writing so that real and
// complex kernels don't differ syntactically.

template <std::floating_point T>
__host__ __device__ static __inline__ auto operator/=(
    CuComplexTrait<T> &c, T val) -> CuComplexTrait<T> & {
    c.x /= val;
    c.y /= val;
    return c;
}

template <std::floating_point T>
__host__ __device__ static __inline__ auto operator/(CuComplexTrait<T> c, T val)
    -> CuComplexTrait<T> {
    return {c.x / val, c.y / val};
}

template <std::floating_point T>
__host__ __device__ static __inline__ auto operator*(CuComplexTrait<T> c, T val)
    -> CuComplexTrait<T> {
    return {c.x * val, c.y * val};
}

template <std::floating_point T>
__host__ __device__ static __inline__ auto operator*(T val, CuComplexTrait<T> c)
    -> CuComplexTrait<T> {
    return {c.x * val, c.y * val};
}

template <CuComplexType C>
__host__ __device__ static __inline__ auto operator-(C c1, C c2) -> C {
    return {c1.x - c2.x, c1.y - c2.y};
}

template <CuComplexType C>
__host__ __device__ static __inline__ auto operator-=(C &c1, C c2) -> C & {
    c1.x -= c2.x;
    c1.y -= c2.y;
    return c1;
}

// Returns indices of the current thread
template <std::size_t Dim>
__device__ auto GetIdxs() -> MultidimIdx<Dim> {
    if constexpr (Dim == 1) {
        return {blockIdx.x * blockDim.x + threadIdx.x};
    } else if constexpr (Dim == 2) {
        return {blockIdx.x * blockDim.x + threadIdx.x,
                blockIdx.y * blockDim.y + threadIdx.y};
    } else if constexpr (Dim == 3) {
        return {blockIdx.x * blockDim.x + threadIdx.x,
                blockIdx.y * blockDim.y + threadIdx.y,
                blockIdx.z * blockDim.z + threadIdx.z};
    }
}

// Returns strides for each dimension
template <std::size_t Dim>
__device__ auto GetStrides() -> std::array<std::size_t, Dim> {
    if constexpr (Dim == 1) {
        return {gridDim.x * blockDim.x};
    } else if constexpr (Dim == 2) {
        return {gridDim.x * blockDim.x, gridDim.y * blockDim.y};
    } else if constexpr (Dim == 3) {
        return {gridDim.x * blockDim.x,
                gridDim.y * blockDim.y,
                gridDim.z * blockDim.z};
    }
}

/**
 * @brief Applies the function \p f to indices in \p Dim nested loops.
 *
 * Each dimensions has:
 * 1. A starting index in \p starts,
 * 2. size in \p ns and
 * 3. a stride (in cast there are too few threads overall) in \p strides.
 */
template <std::size_t Dim, std::size_t CurDim, typename Func>
__device__ void ApplyStrideLoopImpl(MultidimIdx<Dim> &midx,
                                    std::array<std::size_t, Dim> starts,
                                    GridSizes<Dim> ns,
                                    std::array<std::size_t, Dim> strides,
                                    Func f) {
    for (midx[CurDim] = starts[CurDim]; midx[CurDim] < ns[CurDim];
         midx[CurDim] += strides[CurDim]) {
        if constexpr (CurDim > 0) {
            ApplyStrideLoopImpl<Dim, CurDim - 1>(midx, starts, ns, strides, f);
        } else {
            f(midx);
        }
    }
}

/**
 * @brief Applies the function \p f to indices in \p Dim nested loops where each
 * dimension has the size according to \p ns.
 */
template <std::size_t Dim, typename Func>
__device__ void ApplyStrideLoop(GridSizes<Dim> ns, Func &&f) {
    MultidimIdx<Dim> midx{};
    auto starts = GetIdxs<Dim>();
    auto strides = GetStrides<Dim>();
    ApplyStrideLoopImpl<Dim, Dim - 1>(midx, starts, ns, strides, std::forward<Func>(f));
}

template <std::floating_point T, std::size_t Dim>
__device__ auto SumIndexedDenoms(DenomConstPtrs<T, Dim> denoms,
                                 MultidimIdx<Dim> midx,
                                 GridOffsets<Dim> offs) -> T {
    T res = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        res += denoms[d][midx[d] + offs[d]];
    }
    return res;
}

template <std::size_t Dim>
__device__ auto OperatesOnGridOrigin(MultidimIdx<Dim> midx,
                                     GridOffsets<Dim> offs) -> bool {
    bool operates_on_grid_origin{true};
    for (std::size_t d{0}; d < Dim; ++d) {
        operates_on_grid_origin =
            operates_on_grid_origin && (midx[d] + offs[d]) == 0;
    }
    return operates_on_grid_origin;
}

template <std::floating_point T, std::size_t Dim, ValidWorkspace<T> Workspace>
__global__ void DivideByDenoms(Workspace *workspace,
                               GridSizes<Dim> ns,
                               GridOffsets<Dim> offs,
                               DenomConstPtrs<T, Dim> denoms,
                               bool zero_origin) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        if (zero_origin && OperatesOnGridOrigin(midx, offs)) {
            workspace[idx] = Workspace{T{0.0}};
            return;
        }
        workspace[idx] /= SumIndexedDenoms(denoms, midx, offs);
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void UnpackPaddedGrid(T *active, const T * padded, GridSizes<Dim> ns, GridSizes<Dim> ngs) {
    GridSizes<Dim> padded_ns{};
    for (std::size_t d{0}; d < Dim; ++d) {
        padded_ns[d] = ns[d] + 2 * ngs[d];
    }

    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        auto padded_midx = midx;
        for(std::size_t d{0}; d < Dim; ++d) {
            padded_midx[d] += ngs[d];
        }
        active[FlatIndex(midx, ns)] = padded[FlatIndex(padded_midx, padded_ns)];
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void PackPaddedGrid(T * padded, const T * active, GridSizes<Dim> ns, GridSizes<Dim> ngs) {
    GridSizes<Dim> padded_ns{};
    for (std::size_t d{0}; d < Dim; ++d) {
        padded_ns[d] = ns[d] + 2 * ngs[d];
    }

    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        auto padded_midx = midx;
        for(std::size_t d{0}; d < Dim; ++d) {
            padded_midx[d] += ngs[d];
        }
        padded[FlatIndex(padded_midx, padded_ns)] = active[FlatIndex(midx, ns)];
    });
}
}  // namespace poisson_solver::detail::kernels

#endif  // COMMON_CUH
