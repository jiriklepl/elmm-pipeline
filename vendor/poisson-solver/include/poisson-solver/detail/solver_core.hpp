#ifndef SOLVER_CORE_HPP
#define SOLVER_CORE_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/detail/axis_strategies.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/fft_policies.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/reshaper.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>
#include <variant>
#include <vector>

#ifdef POISSON_SOLVER_USE_MPI
#include "poisson-solver/detail/mpi_check.hpp"

#include <mpi.h>
#endif  // POISSON_SOLVER_USE_MPI

namespace poisson_solver::detail {
/**
 * @brief Class encapsulating core solver data.
 * Every solver is composed (at least) of this core.
 *
 * @tparam T Floating point type (only `float` and `double` supported).
 * @tparam Dim Number of grid dimensions.
 * @tparam FFTDim Number of grid dimensions transformed via FFT.
 */
template <std::floating_point T, std::size_t Dim, std::size_t FFTDim = Dim>
    requires(FFTDim <= Dim)
struct SolverCore {
public:
    using HostOneAxisDenoms = std::vector<T>;
    using HostDenoms = std::array<HostOneAxisDenoms, FFTDim>;
    using CudaUniquePtrT = CudaUniquePtr<T>;
    using CudaSharedPtrT = CudaSharedPtr<T>;
    using CudaSharedPtrCmplxT = CudaSharedPtr<CuComplexTrait<T>>;
    using DenomConstPtrs = detail::DenomConstPtrs<T, FFTDim>;

    // FFT needs to know the dimensionality of the whole grid.
    // Transformation of only some dimensions are handled using
    // the 1D and 2D methods of FFTPolicy.
    using CoreFFTPolicy = FFTPolicy<T, Dim>;

    using FFTPlan = typename CoreFFTPolicy::Plan;

    // === Plans ===
    FFTPlan fft_plan;

    // === Grid ===
    GridSizes<Dim> ns;
    GridOffsets<Dim> offs;
    std::size_t cnt;
    std::unique_ptr<IReshaper<T, Dim>> reshaper{};

    // === Physics and Math ===
    std::array<CudaUniquePtrT, FFTDim> cu_denoms;

    // These will be passed into the kernels.
    DenomConstPtrs cu_denom_observers;

    T norm_factor;

    // === Workspaces ===
    // We need two of each to account for out-of-place
    // reshaping in distributed solvers.
    // We ping-pong between the two (i.e., use wspace1, reshape into
    // wspace2, use wspace2, etc.).
    // For more info about workspace allocation see AllocDeviceWorkspaces()
    // and also solver_plan.hpp.
    CudaSharedPtrCmplxT cu_cmplx_workspace1{};
    CudaSharedPtrCmplxT cu_cmplx_workspace2{};
    CudaSharedPtrT cu_real_workspace1{};
    CudaSharedPtrT cu_real_workspace2{};

    // === Device Settings ===
    dim3 cu_block_size;

    /**
     * @brief Constructs core for local solver.
     */
    SolverCore(GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        : SolverCore(BuildLocalBackend(ns, sp),
                     ns,
                     lens,
                     ns,
                     GridOffsets<Dim>{},  // Zero-initialize all elements
                     approx,
                     cuda_block_size,
                     sp) {}

#ifdef POISSON_SOLVER_USE_MPI
    /**
     * @brief Constructs core for distributed solver.
     */
    SolverCore(GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               GridSizes<Dim> gns,
               GridOffsets<Dim> offs,
               MPI_Comm comm,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        : SolverCore(BuildDistributedBackend(ns, offs, sp, comm),
                     ns,
                     lens,
                     gns,
                     offs,
                     approx,
                     cuda_block_size,
                     sp) {}
#endif  // POISSON_SOLVER_USE_MPI

    /**
     * @note The dimensionality of the result is determined by `FFTDim`.
     * The computations in dimensions not transformed via FFT
     * (e.g. Z in `NonuniformZSolver`) contain dependencies in that dimension.
     * This forces the GPU threads to parallelize only on the remaining dims
     * (e.g. X and Y in `NonuniformZSolver`).
     */
    auto ComputeCUDAGridSize(GridSizes<Dim> ns) -> dim3 {
        if constexpr (FFTDim == 1) {
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                    cu_block_size.x};
        } else if constexpr (FFTDim == 2) {
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                        cu_block_size.x,
                    (static_cast<unsigned>(ns[1]) + cu_block_size.y - 1) /
                        cu_block_size.y};
        } else {
            // FFTDim == 3
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                        cu_block_size.x,
                    (static_cast<unsigned>(ns[1]) + cu_block_size.y - 1) /
                        cu_block_size.y,
                    (static_cast<unsigned>(ns[2]) + cu_block_size.z - 1) /
                        cu_block_size.z};
        }
    }

    auto ComputeFullCUDAGridSize(GridSizes<Dim> ns) -> dim3 {
        if constexpr (Dim == 1) {
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                    cu_block_size.x};
        } else if constexpr (Dim == 2) {
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                        cu_block_size.x,
                    (static_cast<unsigned>(ns[1]) + cu_block_size.y - 1) /
                        cu_block_size.y};
        } else {
            // Dim == 3
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                        cu_block_size.x,
                    (static_cast<unsigned>(ns[1]) + cu_block_size.y - 1) /
                        cu_block_size.y,
                    (static_cast<unsigned>(ns[2]) + cu_block_size.z - 1) /
                        cu_block_size.z};
        }
    }

    auto GetBufferTotalSize(GridSizes<Dim> ngs) const -> std::size_t {
        std::size_t total_size{1};
        for(std::size_t d{0}; d < Dim; ++d) {
            total_size *= (ns[d] + 2 * ngs[d]);
        }
        return total_size;
    }

    auto GetRhsPaddedBuffer(GridSizes<Dim> ngs) -> T * {
        if (!cu_padded_rhs.get() || ngs != padded_rhs_ngs) {
            cu_padded_rhs = MakeCudaShared<T>(
                    GetBufferTotalSize(ngs));
            padded_rhs_ngs = ngs;
        }
        return cu_padded_rhs.get();
    }

    auto GetPhiPaddedBuffer(GridSizes<Dim> ngs) -> T * {
        if (!cu_padded_phi.get() || ngs != padded_phi_ngs) {
            auto total_size = GetBufferTotalSize(ngs);
            cu_padded_phi = MakeCudaShared<T>(total_size);
            padded_phi_ngs = ngs;
            CUCH(cudaMemset(cu_padded_phi.get(),
                        0, total_size * sizeof(T)));
        }
        return cu_padded_phi.get();
    }

private:
    // CPU API first copies data here, then uses (un)packing kernels
    // to move it.
    // Users should interact with these buffers only via their getters
    // (those ensure correct size).
    CudaSharedPtrT cu_padded_rhs{};
    CudaSharedPtrT cu_padded_phi{};
    GridSizes<Dim> padded_rhs_ngs{};
    GridSizes<Dim> padded_phi_ngs{};

    struct SolverBackend {
        std::unique_ptr<IReshaper<T, Dim>> reshaper;
        FFTPlan fft_plan;
    };

    /**
     * @brief Master constructor.
     * Initializes everything common for local and distributed solver.
     *
     * @note \p backend should be initialized by the
     * calling ctor and passed here.
     */
    SolverCore(SolverBackend &&backend,
               GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               GridSizes<Dim> gns,
               GridOffsets<Dim> offs,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        : fft_plan(std::move(backend.fft_plan)),
          ns(ns),
          offs(offs),
          cnt(ComputeLocalGridSize(ns)),
          reshaper(std::move(backend.reshaper)),
          norm_factor(ComputeNormFactor(gns, sp)),
          cu_block_size(cuda_block_size) {
        AllocDeviceBuffers(gns,
                           sp.requires_cmplx_wspace,
                           std::holds_alternative<FullLayout>(sp.fft_layout));

        // Denoms are constant for the solver lifetime.
        // They can be copied just once, here.
        const auto denoms = GetEigenvalues(lens, gns, approx, sp);
        CopyDenomsToDevice(denoms, gns);
        cu_denom_observers = CreateDenomObservers();
    }

    static auto BuildLocalBackend(
        GridSizes<Dim> ns, const ISolverPlan<T, FFTDim> &sp) -> SolverBackend {
        // Local solver has all the data and doesn't need any reshaping.
        return {BuildReshaper(DummyReshapePipeline<T, Dim>::From(ns), sp),
                CoreFFTPolicy::Init(ns, sp.fft_layout)};
    }

#ifdef POISSON_SOLVER_USE_MPI
    /**
     * @note The reshaper has to be built before the fft_plan because
     * the fft_plan uses it (in a distributed solver core).
     * However, the fft_plan construction happens in the local/distributed ctor
     * and so we have to build the reshaper alongside it in this method.
     */
    static auto BuildDistributedBackend(GridSizes<Dim> ns,
                                        GridOffsets<Dim> offs,
                                        const ISolverPlan<T, FFTDim> &sp,
                                        MPI_Comm comm) -> SolverBackend {
        auto distributed_reshaper = BuildReshaper(
            ReshapePipeline<T, Dim>::From(
                ns,
                offs,
                comm,
                std::holds_alternative<FullLayout>(sp.fft_layout)),
            sp);

        auto distributed_fft_plan = std::visit(
            Overloaded{[&](FullLayout fl) {
                           return CoreFFTPolicy::Init(ns, offs, comm, fl);
                       },
                       [&](SplitLayout sl) {
                           return CoreFFTPolicy::Init(
                               ns,
                               distributed_reshaper->PencilLayout().ns,
                               distributed_reshaper->SlabLayout().ns,
                               comm,
                               sl);
                       },
                       [](std::monostate) -> FFTPlan {
                           throw std::runtime_error("Visited a monostate.");
                       }},
            sp.fft_layout);

        return {std::move(distributed_reshaper),
                std::move(distributed_fft_plan)};
    }
#endif  // POISSON_SOLVER_USE_MPI

    static auto ComputeLocalGridSize(GridSizes<Dim> ns) -> std::size_t {
        return std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>{});
    }

    static auto BuildReshaper(std::unique_ptr<IReshapePipeline<T, Dim>> &&rp,
                              const ISolverPlan<T, FFTDim> &sp)
        -> std::unique_ptr<IReshaper<T, Dim>> {
        using enum ReshapeType;

        std::size_t pencil_dir{};
        std::array<std::size_t, 2> slab_dirs{};

        // pencil_dir and slab_dirs are left zero-initialized
        // for FullLayout as that utilizes DummyReshaper
        std::visit(
            Overloaded{[&](FullLayout fl) { /* nop */ },
                       [&](SplitLayout sl) {
                           pencil_dir = sl.isolated_dim_idx;
                           slab_dirs[0] = (sl.isolated_dim_idx + 1) % 3;
                           slab_dirs[1] = (sl.isolated_dim_idx + 2) % 3;
                       },
                       [](std::monostate) {
                           throw std::runtime_error("Visited a monostate.");
                       }},
            sp.fft_layout);

        for (const auto &reshape_type : sp.reshape_types) {
            switch (reshape_type) {
                case ToPencils:
                    rp->ToPencils(pencil_dir);
                    break;
                case ToSlabs:
                    rp->ToSlabs(slab_dirs[0], slab_dirs[1]);
                    break;
                default:
                    throw std::runtime_error("Unknown reshape type.");
            }
        }

        return rp->Build();
    }

    auto ComputeNormFactor(GridSizes<Dim> gns,
                           const ISolverPlan<T, FFTDim> &sp) const -> T {
        std::size_t factor = 1;
        for (std::size_t d = 0; d < FFTDim; ++d) {
            const auto &strategy = sp.axis_strategies[d];
            factor *= strategy->ComputeNormalizationFactor(
                gns[d], std::holds_alternative<FullLayout>(sp.fft_layout));
        }
        return static_cast<T>(factor);
    }

    void AllocDeviceWorkspaces(bool requires_cmplx_wspace, bool layout_full) {
        // Workspace Allocation Table
        // rcw : requires_cmplx_wspace
        // lf  : layout_full
        // mpi : POISSON_SOLVER_USE_MPI defined
        // rw1 : cu_real_workspace1,  rw2 : cu_real_workspace2
        // cw1 : cu_cmplx_workspace1, cw2 : cu_cmplx_workspace2

        // aloc : fresh allocation
        //  -   : unallocated

        // +--------------+------+------+------+------+
        // | rcw, lf, mpi | rw1  | rw2  | cw1  | cw2  |
        // +--------------+------+------+------+------+
        // |   0, 0, 0    | aloc | rw1  |  -   |  -   |
        // |   0, 0, 1    | aloc | aloc |  -   |  -   |
        // |   0, 1, 0    | aloc |  -   |  -   |  -   |
        // |   0, 1, 1    | aloc |  -   |  -   |  -   |
        // |   1, 0, 0    | aloc | rw1  | aloc | cw1  |
        // |   1, 0, 1    | aloc | aloc | aloc | aloc |
        // |   1, 1, 0    | aloc |  -   | aloc |  -   |
        // |   1, 1, 1    | aloc |  -   | aloc |  -   |
        // +--------------+------+------+------+------+

        using detail::CuComplexTrait;
        using detail::MakeCudaShared;

#ifdef POISSON_SOLVER_USE_MPI
        auto use_mpi = true;
#else
        auto use_mpi = false;
#endif // POISSON_SOLVER_USE_MPI

        cu_real_workspace1 = MakeCudaShared<T>(reshaper->MaxLocalElements());

        if (requires_cmplx_wspace) {
            cu_cmplx_workspace1 =
                MakeCudaShared<CuComplexTrait<T>>(reshaper->MaxLocalElements());
        }

        if (!layout_full) {
            if (use_mpi) {
                cu_real_workspace2 =
                    MakeCudaShared<T>(reshaper->MaxLocalElements());
                if (requires_cmplx_wspace) {
                    cu_cmplx_workspace2 = MakeCudaShared<CuComplexTrait<T>>(
                        reshaper->MaxLocalElements());
                }
            } else {
                cu_real_workspace2 = cu_real_workspace1;
                cu_cmplx_workspace2 = cu_cmplx_workspace1;
            }
        }
    }

    void AllocDeviceDenoms(GridSizes<Dim> gns) {
        using detail::MakeCudaUnique;
        for (std::size_t d = 0; d < FFTDim; ++d) {
            cu_denoms[d] = MakeCudaUnique<T>(gns[d]);
        }
    }

    void AllocDeviceBuffers(GridSizes<Dim> gns,
                            bool requires_cmplx_wspace,
                            bool layout_full) {
        AllocDeviceWorkspaces(requires_cmplx_wspace, layout_full);
        AllocDeviceDenoms(gns);
    }

    auto GetOneAxisEigenvalues(T len,
                               std::size_t gn,
                               const IApproximation<T> &approx,
                               const IAxisStrategy<T> &axis_strategy) const
        -> HostOneAxisDenoms {
        HostOneAxisDenoms eigenvals(gn);

        const T grid_dx = axis_strategy.GetGridDx(gn, len);
        const T dkx = std::numbers::pi_v<T> * grid_dx / len;
        const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

        for (std::size_t i = 0; i < gn; ++i) {
            eigenvals[i] =
                approx(axis_strategy.GetApproximationInput(i, gn, dkx)) *
                inv_grid_dx_squared;
        }

        return eigenvals;
    }

    auto GetEigenvalues(std::array<T, FFTDim> lens,
                        GridSizes<Dim> gns,
                        const IApproximation<T> &approx,
                        const ISolverPlan<T, FFTDim> &sp) const -> HostDenoms {
        HostDenoms denoms{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            denoms[d] = GetOneAxisEigenvalues(
                lens[d], gns[d], approx, *sp.axis_strategies[d]);
        }
        return denoms;
    }

    /**
     * @warning `cu_denoms` buffers have to be already allocated.
     */
    void CopyDenomsToDevice(const HostDenoms &denoms, GridSizes<Dim> gns) {
        for (std::size_t d = 0; d < FFTDim; ++d) {
            CUCH(cudaMemcpy(cu_denoms[d].get(),
                            denoms[d].data(),
                            gns[d] * sizeof(T),
                            cudaMemcpyHostToDevice));
        }
    }

    /**
     * @warning `cu_denoms` buffers have to be already allocated.
     */
    auto CreateDenomObservers() -> DenomConstPtrs {
        DenomConstPtrs observers{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            observers[d] = cu_denoms[d].get();
        }
        return observers;
    }
};
}  // namespace poisson_solver::detail

#endif  // SOLVER_CORE_HPP
