#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/fft_policies.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/runners/common.hpp"
#include "poisson-solver/detail/runners/common_cmplx.hpp"
#include "poisson-solver/detail/runners/common_real.hpp"
#include "poisson-solver/detail/runners/nonuniform_z.hpp"
#include "poisson-solver/detail/solver_core.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <span>
#include <stdexcept>

#ifdef POISSON_SOLVER_USE_MPI
#include <mpi.h>
#endif  // POISSON_SOLVER_USE_MPI

namespace poisson_solver {
/**
 * struct ExecutionTimes - Aggregate of measured times inside the solver.
 */
struct ExecutionTimes {
    float copy_time_ms = 0;
    float reshape_time_ms = 0;
    float fft_time_ms = 0;
    float kernel_time_ms = 0;

    auto SumTimes() -> float {
        return copy_time_ms + reshape_time_ms + fft_time_ms + kernel_time_ms;
    }
};

/**
 * @brief Abstract solver class for both uniform and nonuniform solvers.
 * Its childer are solvers of the equation \f$ \Delta \phi = b \f$.
 *
 * @tparam T Floating point type (only `float` and `double` supported).
 * @tparam Dim Number of grid dimensions.
 */
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class ISolver {
public:
    virtual ~ISolver() = default;
    /**
     * @brief Solves the equation \f$ \Delta \phi = b \f$.
     *
     * @param[out] phi
     * @param[in] rhs
     * @param[in] ngs_phi Number of ghost cells (on one side)
     * \p phi has in each dimension.
     * @param[in] ngs_rhs Number of ghost cells (on one side)
     * \p rhs has in each dimension.
     */
    virtual ExecutionTimes Execute(std::span<T> phi,
                                   std::span<const T> rhs,
                                   GridSizes<Dim> ngs_phi,
                                   GridSizes<Dim> ngs_rhs) = 0;
    /**
     * @brief Same as Execute() but expects device pointers
     * in \p d_phi and \p d_rhs.
     */
    virtual ExecutionTimes ExecuteGPU(std::span<T> d_phi,
                                      std::span<const T> d_rhs,
                                      GridSizes<Dim> ngs_phi,
                                      GridSizes<Dim> ngs_rhs,
                                      cudaStream_t stream = nullptr) = 0;
    virtual auto GetTotalGridSize() const -> std::size_t = 0;

protected:
    struct CUDAScopedTimer {
        float &accum;
        cudaEvent_t start{};
        cudaEvent_t stop{};

#ifdef POISSON_SOLVER_ENABLE_BENCHMARKS
        explicit CUDAScopedTimer(float &acc)
            : accum(acc) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, detail::execution_stream);
        }

        ~CUDAScopedTimer() {
            cudaEventRecord(stop, detail::execution_stream);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            accum += ms;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
#endif  // POISSON_SOLVER_ENABLE_BENCHMARKS
    };

    struct HybridScopedTimer {
        float &accum;
        std::chrono::time_point<std::chrono::steady_clock> start{};

#ifdef POISSON_SOLVER_ENABLE_BENCHMARKS
        explicit HybridScopedTimer(float &acc)
            : accum(acc) {
            cudaStreamSynchronize(detail::execution_stream);
            start = std::chrono::steady_clock::now();
        }

        ~HybridScopedTimer() {
            cudaStreamSynchronize(detail::execution_stream);
            auto stop = std::chrono::steady_clock::now();
            std::chrono::duration<float, std::milli> ms = stop - start;
            accum += ms.count();
        }
#endif  // POISSON_SOLVER_ENABLE_BENCHMARKS
    };

    ExecutionTimes current_times_;

    static constexpr auto DefaultBlockSize() -> dim3 {
        if constexpr (Dim == 1) {
            return {256};
        } else if constexpr (Dim == 2) {
            return {16, 16};
        } else {
            // Dim == 3
            return {8, 8, 4};
        }
    }
};

/**
 * @brief Solver for a uniform grid.
 */
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class Solver : public ISolver<T, Dim> {
public:
#ifndef POISSON_SOLVER_USE_MPI
    /**
     * @brief Constructs a local poisson solver.
     *
     * @param ns Sizes of the grid in each dimension (without ghost cells).
     * @param lens Real lengths of the grid in each dimension.
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * For more information @see boundary_conditions.hpp.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     *
     * @note The solver assumes that the first index is the fastest changing
     * and the last is the slowest.
     */
    Solver(GridSizes<Dim> ns,
           std::array<T, Dim> lens,
           BCs<Dim> bcs,
           const IApproximation<T> &approx = SpectralApproximation<T>{},
           dim3 cuda_block_size = ISolver<T, Dim>::DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns, lens, approx, cuda_block_size, solver_plan_) {}
#endif  // POISSON_SOLVER_USE_MPI

#ifdef POISSON_SOLVER_USE_MPI
    /**
     * @brief Constructs a distributed poisson solver.
     *
     * @param ns Local sizes of the grid in each dimension (without ghost
     * cells).
     * @param lens Real lengths of the grid in each dimension (global).
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * For more information @see boundary_conditions.hpp.
     * @param gns Global sizes of the grid in each dimension (without ghost
     * cells).
     * @param offs Offset of local data.
     * @param comm The MPI communicator.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     *
     * @note The solver assumes that the first index is the fastest changing
     * and the last is the slowest.
     */
    Solver(GridSizes<Dim> ns,
           std::array<T, Dim> lens,
           BCs<Dim> bcs,
           GridSizes<Dim> gns,
           GridOffsets<Dim> offs,
           MPI_Comm comm,
           const IApproximation<T> &approx = SpectralApproximation<T>{},
           dim3 cuda_block_size = ISolver<T, Dim>::DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns,
                lens,
                gns,
                offs,
                comm,
                approx,
                cuda_block_size,
                solver_plan_) {}
#endif  // POISSON_SOLVER_USE_MPI

    auto Execute(std::span<T> phi,
                 std::span<const T> rhs,
                 GridSizes<Dim> ngs_phi,
                 GridSizes<Dim> ngs_rhs) -> ExecutionTimes override {
        detail::StreamScope stream_scope(nullptr);
#ifndef POISSON_SOLVER_USE_MPI
        detail::FFTPolicy<T, Dim>::SetStream(core_.fft_plan, nullptr);
#endif
        using detail::runners::PackPaddedGrid;
        using detail::runners::UnpackPaddedGrid;

        this->current_times_ = ExecutionTimes{};
        float copy_time{0.0f};
        {
            CUDAScopedTimer t(copy_time);

            auto rhs_padded = core_.GetRhsPaddedBuffer(ngs_rhs);
            CUCH(cudaMemcpy(rhs_padded,
                            rhs.data(),
                            core_.GetBufferTotalSize(ngs_rhs) * sizeof(T),
                            cudaMemcpyHostToDevice));
            UnpackPaddedGrid(core_.cu_real_workspace1.get(),
                             rhs_padded,
                             core_.ns,
                             ngs_rhs,
                             core_.ComputeFullCUDAGridSize(core_.ns),
                             core_.cu_block_size);
        }

        (this->*solver_plan_.solve_method)();

        {
            CUDAScopedTimer t(copy_time);

            auto phi_padded = core_.GetPhiPaddedBuffer(ngs_phi);
            PackPaddedGrid(phi_padded,
                           core_.cu_real_workspace1.get(),
                           core_.ns,
                           ngs_phi,
                           core_.ComputeFullCUDAGridSize(core_.ns),
                           core_.cu_block_size);
            CUCH(cudaMemcpy(phi.data(),
                            phi_padded,
                            core_.GetBufferTotalSize(ngs_phi) * sizeof(T),
                            cudaMemcpyDeviceToHost));
        }
        this->current_times_.copy_time_ms = copy_time;
        return this->current_times_;
    }

    auto ExecuteGPU(std::span<T> d_phi,
                    std::span<const T> d_rhs,
                    GridSizes<Dim> ngs_phi,
                    GridSizes<Dim> ngs_rhs,
                    cudaStream_t stream = nullptr) -> ExecutionTimes override {
        detail::StreamScope stream_scope(stream);
#ifndef POISSON_SOLVER_USE_MPI
        detail::FFTPolicy<T, Dim>::SetStream(core_.fft_plan, stream);
#else
        if (stream) throw std::invalid_argument("nondefault streams require local solver build");
#endif
        using detail::runners::PackPaddedGrid;
        using detail::runners::UnpackPaddedGrid;

        this->current_times_ = ExecutionTimes{};
        float copy_time{0.0f};
        {
            CUDAScopedTimer t(copy_time);

            UnpackPaddedGrid(core_.cu_real_workspace1.get(),
                             d_rhs.data(),
                             core_.ns,
                             ngs_rhs,
                             core_.ComputeFullCUDAGridSize(core_.ns),
                             core_.cu_block_size);
        }

        (this->*solver_plan_.solve_method)();

        {
            CUDAScopedTimer t(copy_time);

            PackPaddedGrid(d_phi.data(),
                           core_.cu_real_workspace1.get(),
                           core_.ns,
                           ngs_phi,
                           core_.ComputeFullCUDAGridSize(core_.ns),
                           core_.cu_block_size);
        }
        this->current_times_.copy_time_ms = copy_time;
        return this->current_times_;
    }

    auto GetTotalGridSize() const -> std::size_t override {
        return core_.cnt;
    }

private:
    friend struct detail::SolverPlan<T, Dim>;
    using FFTPolicy = detail::FFTPolicy<T, Dim>;
    using CUDAScopedTimer = ISolver<T, Dim>::CUDAScopedTimer;
    using HybridScopedTimer = ISolver<T, Dim>::HybridScopedTimer;

    // Plan has to be initialized before core
    detail::SolverPlan<T, Dim> solver_plan_;
    detail::SolverCore<T, Dim> core_;

    void SolveFullPeriodic() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Init(core_.cu_cmplx_workspace1.get(),
                 core_.cu_real_workspace1.get(),
                 core_.ns,
                 core_.ComputeCUDAGridSize(core_.ns),
                 core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward(core_.cu_cmplx_workspace1.get(), core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_cmplx_workspace1.get(),
                           core_.ns,
                           core_.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(core_.ns),
                           core_.cu_block_size,
                           true);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward(core_.cu_cmplx_workspace1.get(),
                                core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.cu_cmplx_workspace1.get(),
                      core_.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(core_.ns),
                      core_.cu_block_size);
        }
    }

    void SolveFullDirichlet() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward(core_.cu_real_workspace1.get(), core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_real_workspace1.get(),
                           core_.ns,
                           core_.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(core_.ns),
                           core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward(core_.cu_real_workspace1.get(), core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(core_.ns),
                      core_.cu_block_size);
        }
    }

    void SolveFullNeumann() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward(core_.cu_real_workspace1.get(), core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_real_workspace1.get(),
                           core_.ns,
                           core_.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(core_.ns),
                           core_.cu_block_size,
                           true);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward(core_.cu_real_workspace1.get(), core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(core_.ns),
                      core_.cu_block_size);
        }
    }

    void SolvePPNs() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        mpi_utils::LocalGridInfo<Dim> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward1D(core_.cu_real_workspace2.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Init(core_.cu_cmplx_workspace1.get(),
                 core_.cu_real_workspace1.get(),
                 new_layout.ns,
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_cmplx_workspace1.get(),
                                 core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_cmplx_workspace1.get(),
                           new_layout.ns,
                           new_layout.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(new_layout.ns),
                           core_.cu_block_size,
                           true);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_cmplx_workspace1.get(),
                                  core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.cu_cmplx_workspace1.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward1D(core_.cu_real_workspace2.get(),
                                  core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }

    void SolvePPD() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        mpi_utils::LocalGridInfo<Dim> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward1D(core_.cu_real_workspace2.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Init(core_.cu_cmplx_workspace1.get(),
                 core_.cu_real_workspace1.get(),
                 new_layout.ns,
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_cmplx_workspace1.get(),
                                 core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_cmplx_workspace1.get(),
                           new_layout.ns,
                           new_layout.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(new_layout.ns),
                           core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_cmplx_workspace1.get(),
                                  core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.cu_cmplx_workspace1.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward1D(core_.cu_real_workspace2.get(),
                                  core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }

    void SolveNsPP() {
        // Same body as PPNs.
        // The only change is in `fft_plan` construction
        // and `reshaper` configuration
        // but this doesn't concern this function.
        SolvePPNs();
    }

    void SolvePNsNs() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        mpi_utils::LocalGridInfo<Dim> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_real_workspace2.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Init(core_.cu_cmplx_workspace1.get(),
                 core_.cu_real_workspace1.get(),
                 new_layout.ns,
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward1D(core_.cu_cmplx_workspace1.get(),
                                 core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_cmplx_workspace1.get(),
                           new_layout.ns,
                           new_layout.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(new_layout.ns),
                           core_.cu_block_size,
                           true);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward1D(core_.cu_cmplx_workspace1.get(),
                                  core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.cu_cmplx_workspace1.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_real_workspace2.get(),
                                  core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }

    void SolvePNsP() {
        // Same body as PPNs.
        // The only change is in `fft_plan` construction
        // and `reshaper` configuration
        // but this doesn't concern this function.
        SolvePPNs();
    }

    void SolveDDP() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        mpi_utils::LocalGridInfo<Dim> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_real_workspace2.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Init(core_.cu_cmplx_workspace1.get(),
                 core_.cu_real_workspace1.get(),
                 new_layout.ns,
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward1D(core_.cu_cmplx_workspace1.get(),
                                 core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_cmplx_workspace1.get(),
                           new_layout.ns,
                           new_layout.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(new_layout.ns),
                           core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward1D(core_.cu_cmplx_workspace1.get(),
                                  core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace1.get(),
                      core_.cu_cmplx_workspace1.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_real_workspace2.get(),
                                  core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }

    void Solve2Real1Real() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        mpi_utils::LocalGridInfo<Dim> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_real_workspace2.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward1D(core_.cu_real_workspace1.get(),
                                 core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            DivideByDenoms(core_.cu_real_workspace1.get(),
                           new_layout.ns,
                           new_layout.offs,
                           core_.cu_denom_observers,
                           core_.ComputeCUDAGridSize(new_layout.ns),
                           core_.cu_block_size,
                           solver_plan_.zero_origin);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward1D(core_.cu_real_workspace1.get(),
                                  core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace2.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_real_workspace2.get(),
                                  core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }

    void Solve1Real2Real() {
        // Same body as 2Real1Real.
        // The only change is in `fft_plan` construction
        // and `reshaper` configuration
        // but this doesn't concern this function.
        Solve2Real1Real();
    }
};

/**
 * @brief Solver for a non-uniform grid.
 *
 * @warning Only works for 3D grids.
 */
template <std::floating_point T>
class NonuniformZSolver : public ISolver<T, 3> {
public:
#ifndef POISSON_SOLVER_USE_MPI
    /**
     * @brief Constructs a local poisson solver with nonuniform Z axis.
     *
     * @param ns Sizes of the grid in each dimension (without ghost cells).
     * @param lens Real lengths of the grid in each dimension.
     * @param z Coordinates representing the cell centers along the Z-axis.
     * Must contain at least `ns[2]` elements.
     * @param z_u Coordinates representing the cell faces along the Z-axis.
     * Must contain at least `ns[2] + 1` elements.
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * For more information @see boundary_conditions.hpp.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     * The `z` component will be unused (always set to 1).
     *
     * @note The solver assumes that the first index is the fastest changing
     * second is middle and third is slowest.
     */
    NonuniformZSolver(
        GridSizes<3> ns,
        std::array<T, 2> lens,
        std::span<const T> z,
        std::span<const T> z_u,
        BCs<3> bcs,
        const IApproximation<T> &approx = SpectralApproximation<T>{},
        dim3 cuda_block_size = DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns, lens, approx, cuda_block_size, solver_plan_) {
        Init(z, z_u, ns);
    }
#endif  // POISSON_SOLVER_USE_MPI

#ifdef POISSON_SOLVER_USE_MPI
    /**
     * @brief Constructs a distributed poisson solver with nonuniform Z axis.
     *
     * @param ns Sizes of the grid in each dimension (without ghost cells).
     * @param lens Real lengths of the grid in each dimension.
     * @param z Coordinates representing the cell centers along the Z-axis.
     * Must contain at least `gns[2]` elements.
     * @param z_u Coordinates representing the cell faces along the Z-axis.
     * Must contain at least `gns[2] + 1` elements.
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * For more information @see boundary_conditions.hpp.
     * @param gns Global sizes of the grid in each dimension (without ghost
     * cells).
     * @param offs Offset of local data.
     * @param comm The MPI communicator.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     * The `z` component will be unused (always set to 1).
     *
     * @note The solver assumes that the first index is the fastest changing
     * second is middle and third is slowest.
     */
    NonuniformZSolver(
        GridSizes<3> ns,
        std::array<T, 2> lens,
        std::span<const T> z,
        std::span<const T> z_u,
        BCs<3> bcs,
        GridSizes<3> gns,
        GridOffsets<3> offs,
        MPI_Comm comm,
        const IApproximation<T> &approx = SpectralApproximation<T>{},
        dim3 cuda_block_size = DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns,
                lens,
                gns,
                offs,
                comm,
                approx,
                cuda_block_size,
                solver_plan_) {
        Init(z, z_u, gns);
    }
#endif  // POISSON_SOLVER_USE_MPI

    ExecutionTimes Execute(std::span<T> phi,
                           std::span<const T> rhs,
                           GridSizes<3> ngs_phi,
                           GridSizes<3> ngs_rhs) override {
        detail::StreamScope stream_scope(nullptr);
#ifndef POISSON_SOLVER_USE_MPI
        detail::FFTPolicy<T, 3>::SetStream(core_.fft_plan, nullptr);
#endif
        using detail::runners::PackPaddedGrid;
        using detail::runners::UnpackPaddedGrid;

        this->current_times_ = ExecutionTimes{};
        float copy_time{0.0f};
        {
            CUDAScopedTimer t(copy_time);

            auto rhs_padded = core_.GetRhsPaddedBuffer(ngs_rhs);
            CUCH(cudaMemcpy(rhs_padded,
                            rhs.data(),
                            core_.GetBufferTotalSize(ngs_rhs) * sizeof(T),
                            cudaMemcpyHostToDevice));
            UnpackPaddedGrid(core_.cu_real_workspace1.get(),
                             rhs_padded,
                             core_.ns,
                             ngs_rhs,
                             core_.ComputeFullCUDAGridSize(core_.ns),
                             core_.cu_block_size);
        }

        (this->*solver_plan_.solve_method)();

        {
            CUDAScopedTimer t(copy_time);

            auto phi_padded = core_.GetPhiPaddedBuffer(ngs_phi);
            PackPaddedGrid(phi_padded,
                           core_.cu_real_workspace1.get(),
                           core_.ns,
                           ngs_phi,
                           core_.ComputeFullCUDAGridSize(core_.ns),
                           core_.cu_block_size);
            CUCH(cudaMemcpy(phi.data(),
                            phi_padded,
                            core_.GetBufferTotalSize(ngs_phi) * sizeof(T),
                            cudaMemcpyDeviceToHost));
        }
        this->current_times_.copy_time_ms = copy_time;
        return this->current_times_;
    }

    ExecutionTimes ExecuteGPU(std::span<T> d_phi,
                              std::span<const T> d_rhs,
                              GridSizes<3> ngs_phi,
                              GridSizes<3> ngs_rhs,
                              cudaStream_t stream = nullptr) override {
        detail::StreamScope stream_scope(stream);
#ifndef POISSON_SOLVER_USE_MPI
        detail::FFTPolicy<T, 3>::SetStream(core_.fft_plan, stream);
#else
        if (stream) throw std::invalid_argument("nondefault streams require local solver build");
#endif
        using detail::runners::PackPaddedGrid;
        using detail::runners::UnpackPaddedGrid;

        this->current_times_ = ExecutionTimes{};
        float copy_time{0.0f};
        {
            CUDAScopedTimer t(copy_time);

            UnpackPaddedGrid(core_.cu_real_workspace1.get(),
                             d_rhs.data(),
                             core_.ns,
                             ngs_rhs,
                             core_.ComputeFullCUDAGridSize(core_.ns),
                             core_.cu_block_size);
        }

        (this->*solver_plan_.solve_method)();

        {
            CUDAScopedTimer t(copy_time);

            PackPaddedGrid(d_phi.data(),
                           core_.cu_real_workspace1.get(),
                           core_.ns,
                           ngs_phi,
                           core_.ComputeFullCUDAGridSize(core_.ns),
                           core_.cu_block_size);
        }
        this->current_times_.copy_time_ms = copy_time;
        return this->current_times_;
    }

    auto GetTotalGridSize() const -> std::size_t override {
        return core_.cnt;
    }

private:
    friend struct detail::NonuniformZSolverPlan<T>;
    using CUDAScopedTimer = ISolver<T, 3>::CUDAScopedTimer;
    using HybridScopedTimer = ISolver<T, 3>::HybridScopedTimer;

    // Plan has to be initialized before core
    detail::NonuniformZSolverPlan<T> solver_plan_;
    detail::SolverCore<T, 3, 2> core_;

    using FFTPolicy = decltype(core_)::CoreFFTPolicy;
    using CudaUniquePtrT = decltype(core_)::CudaUniquePtrT;

    using TridiagMatrixDiagonal = std::vector<T>;
    using TridiagMatrix = std::array<TridiagMatrixDiagonal, 3>;

    CudaUniquePtrT cu_mat_a_;
    CudaUniquePtrT cu_mat_b_;
    CudaUniquePtrT cu_mat_c_;

    CudaUniquePtrT cu_cp_workspace_;

    void Init(std::span<const T> z, std::span<const T> z_u, GridSizes<3> gns) {
        core_.cu_block_size.z = 1;

        auto gnz = gns[2];
        AllocateDeviceBuffers(gnz);
        auto mat = BuildTridiagMatrix(z, z_u, gnz);
        CopyTridiagMatrixToDevice(mat, gnz);
    }

    /**
     * @note Shadows the `DefaultBlockSize` method in `ISolver` because
     * `NonuniformZSolver` doesn't parallelize in Z dimension.
     *
     * @return Default block size for `NonuniformZSolver`.
     */
    static constexpr auto DefaultBlockSize() -> dim3 {
        return {16, 16, 1};
    }

    void AllocateDeviceBuffers(std::size_t gnz) {
        using detail::MakeCudaUnique;

        cu_mat_a_ = MakeCudaUnique<T>(gnz);
        cu_mat_b_ = MakeCudaUnique<T>(gnz);
        cu_mat_c_ = MakeCudaUnique<T>(gnz);
        cu_cp_workspace_ =
            MakeCudaUnique<T>(core_.reshaper->MaxLocalElements());
    }

    static auto BuildMatrixDiagonalA(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t gnz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_a(gnz, 0);
        for (std::size_t i = 1; i < gnz; ++i) {
            mat_a[i] = 1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_a;
    }

    static auto BuildMatrixDiagonalB(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t gnz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_b(gnz, 0);
        if (gnz == 1) {
            mat_b[0] = 0;
        } else {
            mat_b[0] = -1.0 / (z[1] - z[0]) / (z_u[1] - z_u[0]);
        }

        for (std::size_t i = 1; i < gnz - 1; ++i) {
            mat_b[i] = -1.0 / (z[i + 1] - z[i]) / (z_u[i + 1] - z_u[i]);
            mat_b[i] -= 1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }

        if (gnz > 1) {
            std::size_t i = gnz - 1;
            mat_b[i] = -1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_b;
    }

    static auto BuildMatrixDiagonalC(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t gnz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_c(gnz, 0);
        for (std::size_t i = 0; i < gnz - 1; ++i) {
            mat_c[i] = 1.0 / (z[i + 1] - z[i]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_c;
    }

    static auto BuildTridiagMatrix(std::span<const T> z,
                                   std::span<const T> z_u,
                                   std::size_t gnz) -> TridiagMatrix {
        if (z.size() < gnz) {
            throw std::invalid_argument("The size of `z` is too small.");
        }
        if (z_u.size() < gnz + 1) {
            throw std::invalid_argument("The size of `z_u` is too small.");
        }

        return {BuildMatrixDiagonalA(z, z_u, gnz),
                BuildMatrixDiagonalB(z, z_u, gnz),
                BuildMatrixDiagonalC(z, z_u, gnz)};
    }

    /**
     * @brief Copies all three diagonals to the device.
     *
     * @param[in] mat The three diagonals (all allocated to have size \p nz).
     * @param[in] nz Main diagonal size.
     */
    void CopyTridiagMatrixToDevice(TridiagMatrix mat, std::size_t gnz) {
        CUCH(cudaMemcpy(cu_mat_a_.get(),
                        mat[0].data(),
                        gnz * sizeof(T),
                        cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_mat_b_.get(),
                        mat[1].data(),
                        gnz * sizeof(T),
                        cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_mat_c_.get(),
                        mat[2].data(),
                        gnz * sizeof(T),
                        cudaMemcpyHostToDevice));
    }

    void SolvePPNs() {
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;
        using detail::runners::nonuniform_z::TridiagZ;

        mpi_utils::LocalGridInfo<3> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Init(core_.cu_cmplx_workspace1.get(),
                 core_.cu_real_workspace2.get(),
                 new_layout.ns,
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_cmplx_workspace1.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout =
                core_.reshaper->ReshapeCmplx(core_.cu_cmplx_workspace2.get(),
                                             core_.cu_cmplx_workspace1.get());
        }


        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);
            TridiagZ(core_.cu_cmplx_workspace2.get(),
                     cu_cp_workspace_.get(),
                     new_layout.ns,
                     {new_layout.offs[0], new_layout.offs[1]},
                     core_.cu_denom_observers,
                     cu_mat_a_.get(),
                     cu_mat_b_.get(),
                     cu_mat_c_.get(),
                     core_.ComputeCUDAGridSize(new_layout.ns),
                     core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout =
                core_.reshaper->ReshapeCmplx(core_.cu_cmplx_workspace1.get(),
                                             core_.cu_cmplx_workspace2.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_cmplx_workspace1.get(),
                                  core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace2.get(),
                      core_.cu_cmplx_workspace1.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }

    void SolveFullNeumann() {
        using detail::runners::common_real::Normalize;
        using detail::runners::nonuniform_z::TridiagZ;

        mpi_utils::LocalGridInfo<3> new_layout;

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Forward2D(core_.cu_real_workspace2.get(),
                                 core_.fft_plan);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);
            TridiagZ(core_.cu_real_workspace1.get(),
                     cu_cp_workspace_.get(),
                     new_layout.ns,
                     {new_layout.offs[0], new_layout.offs[1]},
                     core_.cu_denom_observers,
                     cu_mat_a_.get(),
                     cu_mat_b_.get(),
                     cu_mat_c_.get(),
                     core_.ComputeCUDAGridSize(new_layout.ns),
                     core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            new_layout = core_.reshaper->ReshapeReal(
                core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        }

        {
            HybridScopedTimer t(this->current_times_.fft_time_ms);

            FFTPolicy::Backward2D(core_.cu_real_workspace2.get(),
                                  core_.fft_plan);
        }

        {
            CUDAScopedTimer t(this->current_times_.kernel_time_ms);

            Normalize(core_.cu_real_workspace2.get(),
                      new_layout.ns,
                      core_.norm_factor,
                      core_.ComputeCUDAGridSize(new_layout.ns),
                      core_.cu_block_size);
        }

        {
            HybridScopedTimer t(this->current_times_.reshape_time_ms);

            core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                        core_.cu_real_workspace2.get());
        }
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
