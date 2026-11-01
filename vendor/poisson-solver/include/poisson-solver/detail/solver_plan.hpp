#ifndef SOLVER_PLAN_HPP
#define SOLVER_PLAN_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/axis_strategies.hpp"
#include "poisson-solver/detail/fft_policies.hpp"
#include "poisson-solver/detail/reshaper.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace poisson_solver {
// === Forward Declarations ===
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class Solver;

template <std::floating_point T>
class NonuniformZSolver;
}  // namespace poisson_solver

namespace poisson_solver::detail {

// === Structs ===

/**
 * struct ISolverPlan - Holds the behavior and configuration of
 * the solver that is dependent on BCs.
 *
 * Also performs verification of BC combinations during instantiation.
 * Doesn't manage any buffers.
 *
 * @tparam T Float type
 * @tparam FFTDim Number of grid dimensions transformed via FFT.
 */
template <std::floating_point T, std::size_t FFTDim>
    requires(FFTDim <= 3 && FFTDim >= 1)
struct ISolverPlan {
public:
    using AxisStrategies =
        std::array<std::unique_ptr<const IAxisStrategy<T>>, FFTDim>;

    AxisStrategies axis_strategies{};
    VariantLayout fft_layout{};
    bool requires_cmplx_wspace{true};

    // The derived classes will append to this as part of their ctors.
    std::vector<ReshapeType> reshape_types{};

protected:
    ISolverPlan(BCs<FFTDim> bcs, VariantLayout default_layout)
        : axis_strategies(GetAxisStrategies(bcs)),
          fft_layout(default_layout) {}

    auto GetAxisFFTType(BCPair bcp) -> FFTTransformType {
        using enum FFTTransformType;

        if (IsPeriodic(bcp)) {
            return Complex;
        }
        if (IsDirichlet(bcp)) {
            return DST1;
        }
        if (IsDirichletStag(bcp)) {
            return DST2;
        }
        if (IsNeumann(bcp)) {
            return DCT1;
        }
        if (IsNeumannStag(bcp)) {
            return DCT2;
        }
        if (IsDirichletNeumann(bcp)) {
            return DST3;
        }
        if (IsNeumannDirichlet(bcp)) {
            return DCT3;
        }
        if (IsDirichletStagNeumannStag(bcp)) {
            return DST4;
        }
        if (IsNeumannStagDirichletStag(bcp)) {
            return DCT4;
        }
        throw std::invalid_argument("Invalid combination of 1D BCs.");
    }

private:
    static auto GetAxisStrategy(BCPair bcp)
        -> std::unique_ptr<IAxisStrategy<T>> {
        if (IsPeriodic(bcp)) {
            return std::make_unique<PeriodicStrategy<T>>();
        }
        if (IsDirichlet(bcp)) {
            return std::make_unique<DirichletStrategy<T>>();
        }
        if (IsNeumann(bcp)) {
            return std::make_unique<NeumannStrategy<T>>();
        }
        if (IsDirichletStag(bcp)) {
            return std::make_unique<DirichletStagStrategy<T>>();
        }
        if (IsNeumannStag(bcp)) {
            return std::make_unique<NeumannStagStrategy<T>>();
        }
        if (IsDirichletNeumann(bcp) || IsNeumannDirichlet(bcp) ||
            IsDirichletStagNeumannStag(bcp) ||
            IsNeumannStagDirichletStag(bcp)) {
            return std::make_unique<MixedStrategy<T>>();
        }
        throw std::invalid_argument("Invalid boundary condition combination.");
    }

    static auto GetAxisStrategies(BCs<FFTDim> bcs) -> AxisStrategies {
        AxisStrategies strats{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            strats[d] = GetAxisStrategy(bcs[d]);
        }
        return strats;
    }
};

template <std::floating_point T, std::size_t FFTDim>
struct SolverPlan : public ISolverPlan<T, FFTDim> {
public:
    using SolveMethod = void (poisson_solver::Solver<T, FFTDim>::*)();

    SolveMethod solve_method{};
    bool zero_origin{false};

    SolverPlan(BCs<FFTDim> bcs)
        : ISolverPlan<T, FFTDim>(bcs, FullLayout{}) {
#ifdef POISSON_SOLVER_USE_MPI
        if constexpr (FFTDim == 3) {
            if (TryConstructFromFullRealBCsSplit(bcs)) {
                return;
            }
        }
#endif  // POISSON_SOLVER_USE_MPI
        bool all_bcs_identical = TryConstructFromFullBCs(bcs);
        if (all_bcs_identical) {
            return;
        }
        ConstructFromNonFullBCs(bcs);
    }

private:
#ifdef POISSON_SOLVER_USE_MPI
    /**
     * @brief Tries to construct the object for full real BCs
     * in a distributed 3D solver.
     *
     * heFFTe for some reason calls the pre/pos_forward/backward_kernel
     * (kernels that restructure data before/after FFT so that the FFT
     * computes the desired DCT/DST) so many times that it slows
     * the solver down.
     * We instead use split layouts and do the transforms ourselves
     * with the heFFTe MPI API.
     *
     * @note This issue doesn't affect full periodic solvers and
     * therefore they remain on the full layout path.
     */
    auto TryConstructFromFullRealBCsSplit(BCs<3> bcs) -> bool {
        using std::ranges::all_of;
        using std::ranges::none_of;
        using enum ReshapeType;

        bool is_real_full =
            all_of(bcs, IsDirichlet) || all_of(bcs, IsDirichletStag) ||
            all_of(bcs, IsNeumann) || all_of(bcs, IsNeumannStag);
        if (!is_real_full) {
            return false;
        }

        zero_origin = none_of(bcs, HasDirichletOrStag);

        this->fft_layout = SplitLayout{};
        auto &layout = std::get<SplitLayout>(this->fft_layout);

        auto forward_type = this->GetAxisFFTType(bcs[0]);
        layout.forward_type_1d = forward_type;
        layout.forward_type_2d = forward_type;
        layout.isolated_dim_idx = 2;

        solve_method = &Solver<T, FFTDim>::Solve2Real1Real;
        this->reshape_types.emplace_back(ToSlabs);
        this->reshape_types.emplace_back(ToPencils);
        this->requires_cmplx_wspace = false;

        return true;
    }
#endif  // POISSON_SOLVER_USE_MPI

    /**
     * @brief Tries to construct the object in case all BCs are identical
     * (this behavior is shared among all solver dimensionalities).
     */
    auto TryConstructFromFullBCs(BCs<FFTDim> bcs) -> bool {
        using std::ranges::all_of;
        using enum FFTTransformType;

        auto &layout = std::get<FullLayout>(this->fft_layout);

        if (all_of(bcs, IsPeriodic)) {
            solve_method = &Solver<T, FFTDim>::SolveFullPeriodic;
            layout.forward_type = Complex;
        } else if (all_of(bcs, IsDirichlet)) {
            solve_method = &Solver<T, FFTDim>::SolveFullDirichlet;
            layout.forward_type = DST1;
            this->requires_cmplx_wspace = false;
        } else if (all_of(bcs, IsDirichletStag)) {
            solve_method = &Solver<T, FFTDim>::SolveFullDirichlet;
            layout.forward_type = DST2;
            this->requires_cmplx_wspace = false;
        } else if (all_of(bcs, IsNeumann)) {
            solve_method = &Solver<T, FFTDim>::SolveFullNeumann;
            layout.forward_type = DCT1;
            this->requires_cmplx_wspace = false;
        } else if (all_of(bcs, IsNeumannStag)) {
            solve_method = &Solver<T, FFTDim>::SolveFullNeumann;
            layout.forward_type = DCT2;
            this->requires_cmplx_wspace = false;
        } else {
            return false;
        }
        return true;
    }

    void ConstructFromNonFullBCs1D(BCs<1> bcs) {
        using enum FFTTransformType;

        auto &layout = std::get<FullLayout>(this->fft_layout);
        BCPair bcp = bcs[0];

        // The function can also return FFT types for full bcs but
        // as the warning above this function states, that shouldn't happen
        layout.forward_type = this->GetAxisFFTType(bcp);

        // In all non-throwing cases the solve method is full dirichlet
        // and we don't need a complex wspace
        solve_method = &Solver<T, FFTDim>::SolveFullDirichlet;
        this->requires_cmplx_wspace = false;
    }

    void ConstructFromNonFullBCs3D(BCs<3> bcs) {
        using std::ranges::any_of;
        using std::ranges::none_of;
        using enum FFTTransformType;
        using enum ReshapeType;

        // Non-full 3D case => layout must be split.
        this->fft_layout = SplitLayout{};

        auto &layout = std::get<SplitLayout>(this->fft_layout);
        auto [bcx, bcy, bcz] = bcs;

        if (IsPeriodic(bcx) && IsPeriodic(bcy) && IsNeumannOrStag(bcz)) {
            solve_method = &Solver<T, FFTDim>::SolvePPNs;
            layout.forward_type_1d = this->GetAxisFFTType(bcz);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 2;
            this->reshape_types.emplace_back(ToPencils);
            this->reshape_types.emplace_back(ToSlabs);

        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && !HasPeriodic(bcz) &&
                   HasDirichletOrStag(bcz)) {
            solve_method = &Solver<T, FFTDim>::SolvePPD;
            layout.forward_type_1d = this->GetAxisFFTType(bcz);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 2;
            this->reshape_types.emplace_back(ToPencils);
            this->reshape_types.emplace_back(ToSlabs);

        } else if (IsNeumannOrStag(bcx) && IsPeriodic(bcy) && IsPeriodic(bcz)) {
            solve_method = &Solver<T, FFTDim>::SolveNsPP;
            layout.forward_type_1d = this->GetAxisFFTType(bcx);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 0;
            this->reshape_types.emplace_back(ToPencils);
            this->reshape_types.emplace_back(ToSlabs);

        } else if (IsPeriodic(bcx) &&
                   ((IsNeumann(bcy) && IsNeumann(bcz)) ||
                    (IsNeumannStag(bcy) && IsNeumannStag(bcz)))) {
            solve_method = &Solver<T, FFTDim>::SolvePNsNs;
            layout.forward_type_1d = Complex;
            layout.forward_type_2d = this->GetAxisFFTType(bcy);
            layout.isolated_dim_idx = 0;
            this->reshape_types.emplace_back(ToSlabs);
            this->reshape_types.emplace_back(ToPencils);

        } else if (IsPeriodic(bcx) && IsNeumannOrStag(bcy) && IsPeriodic(bcz)) {
            solve_method = &Solver<T, FFTDim>::SolvePNsP;
            layout.forward_type_1d = this->GetAxisFFTType(bcy);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 1;
            this->reshape_types.emplace_back(ToPencils);
            this->reshape_types.emplace_back(ToSlabs);

        } else if (((IsDirichlet(bcx) && IsDirichlet(bcy)) ||
                    (IsDirichletStag(bcx) && IsDirichletStag(bcy))) &&
                   IsPeriodic(bcz)) {
            solve_method = &Solver<T, FFTDim>::SolveDDP;
            layout.forward_type_1d = Complex;
            layout.forward_type_2d = this->GetAxisFFTType(bcx);
            layout.isolated_dim_idx = 2;
            this->reshape_types.emplace_back(ToSlabs);
            this->reshape_types.emplace_back(ToPencils);

        } else if (none_of(bcs, IsPeriodic) && (bcx == bcy) && (bcx != bcz) &&
                   any_of(bcs, HasDirichletOrStag)) {
            solve_method = &Solver<T, FFTDim>::Solve2Real1Real;
            layout.forward_type_1d = this->GetAxisFFTType(bcz);
            layout.forward_type_2d = this->GetAxisFFTType(bcx);
            layout.isolated_dim_idx = 2;
            this->reshape_types.emplace_back(ToSlabs);
            this->reshape_types.emplace_back(ToPencils);
            this->requires_cmplx_wspace = false;

        } else if (none_of(bcs, IsPeriodic) && (bcy == bcz) && (bcx != bcy) &&
                   any_of(bcs, HasDirichletOrStag)) {
            solve_method = &Solver<T, FFTDim>::Solve1Real2Real;
            layout.forward_type_1d = this->GetAxisFFTType(bcx);
            layout.forward_type_2d = this->GetAxisFFTType(bcy);
            layout.isolated_dim_idx = 0;
            this->reshape_types.emplace_back(ToSlabs);
            this->reshape_types.emplace_back(ToPencils);
            this->requires_cmplx_wspace = false;

        } else {
            throw std::invalid_argument("Invalid combination of 3D BCs.");
        }
    }

    /**
     * @brief Fallback option if all BCs are not identical.
     *
     * @warning `TryConstructFromFullBCs` should be called first and only in
     * case of its failure should we call this function.
     */
    void ConstructFromNonFullBCs(BCs<FFTDim> bcs) {
        if constexpr (FFTDim == 1) {
            ConstructFromNonFullBCs1D(bcs);
        } else if constexpr (FFTDim == 2) {
            // In 2D it is only valid to use full BCs.
            throw std::invalid_argument("Invalid combination of 2D BCs.");
        } else if constexpr (FFTDim == 3) {
            ConstructFromNonFullBCs3D(bcs);
        }
    }
};

/**
 * struct NonuniformZSolverPlan - Plan for a solver nonuniform in Z axis.
 *
 * @tparam T Floating point type.
 */
template <std::floating_point T>
struct NonuniformZSolverPlan : public ISolverPlan<T, 2> {
public:
    using SolveMethod = void (poisson_solver::NonuniformZSolver<T>::*)();

    SolveMethod solve_method{};

    /**
     * @note All three BCs are required to verify their correctness for
     * nonuniform Z solver.
     */
    explicit NonuniformZSolverPlan(BCs<3> bcs)
        : ISolverPlan<T, 2>(BCs<2>{bcs[0], bcs[1]}, SplitLayout{}) {
        using std::ranges::all_of;
        using enum FFTTransformType;
        using enum ReshapeType;

        auto &layout = std::get<SplitLayout>(this->fft_layout);
        auto [bcx, bcy, bcz] = bcs;

        layout.isolated_dim_idx = 2;
        this->reshape_types.emplace_back(ToSlabs);
        this->reshape_types.emplace_back(ToPencils);

        if (all_of(bcs, IsNeumann) || all_of(bcs, IsNeumannStag)) {
            solve_method = &NonuniformZSolver<T>::SolveFullNeumann;
            layout.forward_type_2d = this->GetAxisFFTType(bcx);
            this->requires_cmplx_wspace = false;
        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && IsNeumannOrStag(bcz)) {
            solve_method = &NonuniformZSolver<T>::SolvePPNs;
            layout.forward_type_2d = Complex;
        } else {
            throw std::invalid_argument(
                "Invalid combination of nonuniform BCs.");
        }
    }
};
}  // namespace poisson_solver::detail

#endif  // SOLVER_PLAN_HPP
