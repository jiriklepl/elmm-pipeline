#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/execution.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>

#ifdef POISSON_SOLVER_USE_MPI
#include <heffte.h>
#endif  // POISSON_SOLVER_USE_MPI

namespace poisson_solver::detail {
// === Structs ===
// Helper struct for `std::variant` visiting.
template <typename... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};

struct CudaFreeDeleter {
    void operator()(void *ptr) {
        DeviceFree(ptr);
    }
};

/**
 * struct CuComplexTraitImpl - Helper that maps the real type to
 * CUDA complex type.
 */
template <std::floating_point T>
struct CuComplexTraitImpl;

template <>
struct CuComplexTraitImpl<float> {
    using Type = cuComplex;
};

template <>
struct CuComplexTraitImpl<double> {
    using Type = cuDoubleComplex;
};

// === Aliases ===
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaFreeDeleter>;

template <typename T>
using CudaSharedPtr = std::shared_ptr<T>;

template <std::floating_point T>
using CuComplexTrait = typename CuComplexTraitImpl<T>::Type;

template <std::size_t Dim>
using MultidimIdx = std::array<std::size_t, Dim>;

template <std::floating_point T, std::size_t Dim>
using DenomConstPtrs = std::array<const T *, Dim>;

// === Concepts ===
// Restricts workspace to be a real or complex workspace
template <typename Workspace, typename T>
concept ValidWorkspace =
    std::floating_point<T> && (std::is_same_v<Workspace, T> ||
                               std::is_same_v<Workspace, CuComplexTrait<T>>);

// === Functions ===
template <std::size_t Dim>
constexpr auto FlatIndex(MultidimIdx<Dim> midx,
                         GridSizes<Dim> ns) -> std::size_t {
    std::size_t idx = 0;
    std::size_t stride = 1;

    for (std::size_t i = 0; i < Dim; ++i) {
        idx += midx[i] * stride;
        stride *= ns[i];
    }

    return idx;
}

/**
 * @brief Creates a unique pointer to memory
 * such that it can hold \p size elements of type \p T.
 */
template <typename T>
auto MakeCudaUnique(std::size_t size) -> CudaUniquePtr<T> {
    T *ptr;
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::overflow_error("device allocation overflow");
    CUCH(DeviceAllocate(reinterpret_cast<void **>(&ptr), size * sizeof(T)));
    return CudaUniquePtr<T>(ptr);
}

/**
 * @brief Creates a shared pointer to memory
 * such that it can hold \p size elements of type \p T.
 */
template <typename T>
auto MakeCudaShared(std::size_t size) -> CudaSharedPtr<T> {
    T *ptr;
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::overflow_error("device allocation overflow");
    CUCH(DeviceAllocate(reinterpret_cast<void **>(&ptr), size * sizeof(T)));
    return CudaSharedPtr<T>(ptr, CudaFreeDeleter{});
}

/**
 * @brief Transfers the centered active grid between a padded buffer and an active buffer.
 */
template <typename T, std::size_t Dim>
void MemcpyActiveGrid(T *padded_ptr,
                      T *active_ptr,
                      GridSizes<Dim> active_ns,
                      GridSizes<Dim> ngs,
                      cudaMemcpyKind kind,
                      bool padded_to_active) {
    cudaMemcpy3DParms params{};

    // At multiple places in this function, we compute
    // sizes for all three dims but only the first (X)
    // size is multiplied by `sizeof(T)`.
    // This is because the Y and Z sizes are then expressed
    // in terms of the X size which is already in bytes.

    GridSizes<Dim> full_ns;
    for (std::size_t d = 0; d < Dim; ++d) {
        full_ns[d] = active_ns[d] + 2 * ngs[d];
    }

    // Extent of the active grid (the sub-volume to copy)
    params.extent = make_cudaExtent(active_ns[0] * sizeof(T),
                                    (Dim >= 2) ? active_ns[1] : 1,
                                    (Dim >= 3) ? active_ns[2] : 1);

    // Padded side
    cudaPitchedPtr padded_pitched =
        make_cudaPitchedPtr(padded_ptr,
                            full_ns[0] * sizeof(T),
                            full_ns[0],
                            (Dim >= 2) ? full_ns[1] : 1);

    // Active side (contiguous)
    cudaPitchedPtr active_pitched =
        make_cudaPitchedPtr(active_ptr,
                            active_ns[0] * sizeof(T),
                            active_ns[0],
                            (Dim >= 2) ? active_ns[1] : 1);

    std::size_t offset_x = ngs[0];
    std::size_t offset_y = (Dim >= 2) ? ngs[1] : 0;
    std::size_t offset_z = (Dim >= 3) ? ngs[2] : 0;

    cudaPos padded_pos = make_cudaPos(offset_x * sizeof(T), offset_y, offset_z);
    cudaPos active_pos = make_cudaPos(0, 0, 0);

    if (padded_to_active) {
        params.srcPtr = padded_pitched;
        params.srcPos = padded_pos;
        params.dstPtr = active_pitched;
        params.dstPos = active_pos;
    } else {
        params.srcPtr = active_pitched;
        params.srcPos = active_pos;
        params.dstPtr = padded_pitched;
        params.dstPos = padded_pos;
    }

    params.kind = kind;
    CUCH(cudaMemcpy3D(&params));
}

#ifdef POISSON_SOLVER_USE_MPI
/**
 * @brief Map from \p ns and \p offs to heffte::box3d<>.
 */
template <std::size_t Dim>
auto GetGridBox(GridSizes<Dim> ns, GridOffsets<Dim> offs) {
    auto begin_x = offs[0];
    auto begin_y = (Dim >= 2) ? offs[1] : 0;
    auto begin_z = (Dim >= 3) ? offs[2] : 0;

    auto end_x = begin_x + ns[0] - 1;
    auto end_y = (Dim >= 2) ? (begin_y + ns[1] - 1) : 0;
    auto end_z = (Dim >= 3) ? (begin_z + ns[2] - 1) : 0;

    std::array<int, 3> low = {static_cast<int>(begin_x),
                              static_cast<int>(begin_y),
                              static_cast<int>(begin_z)};
    std::array<int, 3> high = {static_cast<int>(end_x),
                               static_cast<int>(end_y),
                               static_cast<int>(end_z)};

    return heffte::box3d<>{low, high};
}
#endif  // POISSON_SOLVER_USE_MPI
}  // namespace poisson_solver::detail

#endif  // HELPERS_HPP
