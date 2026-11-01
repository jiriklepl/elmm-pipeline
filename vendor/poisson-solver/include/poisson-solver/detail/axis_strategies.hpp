#ifndef AXIS_STRATEGIES_HPP
#define AXIS_STRATEGIES_HPP

#include <concepts>
#include <cstddef>

/**
 * @file
 *
 * This file groups together certain computations based on boundary
 * conditions of ONE axis. Valid boundary conditions on an axis are explicitly
 * enumerated in the `ISolverPlan::GetAxisStrategy` function.
 */

namespace poisson_solver::detail::axis_strategy_internals {
/**
 * HeFFTe is used as an MPI FFT library for full solvers
 * and its unscaled results do not match those
 * of VkFFT (the local FFT library).
 * It is therefore necessary to adjust the norm_factor in those cases.
 *
 * @param norm_factor
 * @param times Multiplied by this.
 * @param solver_full If all BCs are identical.
 * @return Adjusted norm factor.
 */
inline auto AdjustNormFactorIfHeFFTeUsed(std::size_t norm_factor,
                                         [[maybe_unused]] std::size_t times,
                                         [[maybe_unused]] bool solver_full)
    -> std::size_t {
#ifdef POISSON_SOLVER_USE_MPI
    return solver_full ? norm_factor * times : norm_factor;
#else
    return norm_factor;
#endif  // POISSON_SOLVER_USE_MPI
}
}  // namespace poisson_solver::detail::axis_strategy_internals

namespace poisson_solver::detail {
template <std::floating_point T>
class IAxisStrategy {
public:
    virtual auto GetGridDx(std::size_t gn, T len) const -> T = 0;
    virtual auto GetApproximationInput(std::size_t gi,
                                       std::size_t gn,
                                       T dkx) const -> T = 0;
    virtual auto ComputeNormalizationFactor(
        std::size_t gn,
        [[maybe_unused]] bool solver_full) const -> std::size_t = 0;

    virtual ~IAxisStrategy() = default;
};

template <std::floating_point T>
class PeriodicStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t gn, T len) const -> T override {
        return len / gn;
    }

    auto GetApproximationInput(std::size_t gi,
                               std::size_t gn,
                               T dkx) const -> T override {
        return gi <= (gn / 2) ? gi * dkx : (gn - gi) * dkx;
    }

    auto ComputeNormalizationFactor(std::size_t gn,
                                    [[maybe_unused]] bool solver_full) const
        -> std::size_t override {
        return gn;
    }
};

template <std::floating_point T>
class DirichletStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t gn, T len) const -> T override {
        return len / (gn + 1);
    }

    auto GetApproximationInput(std::size_t gi,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        return (gi + 1) * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t gn,
                                    [[maybe_unused]] bool solver_full) const
        -> std::size_t override {
        // HeFFTe is used in a modified version where
        // adjusting this isn't required
        return 2 * (gn + 1);
    }
};

template <std::floating_point T>
class NeumannStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t gn, T len) const -> T override {
        return len / (gn - 1);
    }

    auto GetApproximationInput(std::size_t gi,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        return gi * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t gn, bool solver_full) const
        -> std::size_t override {
        // HeFFTe is used in a modified version where
        // adjusting this isn't required
        return 2 * (gn - 1);
    }
};

template <std::floating_point T>
class DirichletStagStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t gn, T len) const -> T override {
        return len / gn;
    }

    auto GetApproximationInput(std::size_t gi,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        return (gi + 1) * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t gn, bool solver_full) const
        -> std::size_t override {
        using axis_strategy_internals::AdjustNormFactorIfHeFFTeUsed;
        return AdjustNormFactorIfHeFFTeUsed(2 * gn, 2, solver_full);
    }
};

template <std::floating_point T>
class NeumannStagStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t gn, T len) const -> T override {
        return len / gn;
    }

    auto GetApproximationInput(std::size_t gi,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        return gi * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t gn,
                                    [[maybe_unused]] bool solver_full) const
        -> std::size_t override {
        using axis_strategy_internals::AdjustNormFactorIfHeFFTeUsed;
        return AdjustNormFactorIfHeFFTeUsed(2 * gn, 2, solver_full);
    }
};

template <std::floating_point T>
class MixedStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t gn, T len) const -> T override {
        return len / gn;
    }

    auto GetApproximationInput(std::size_t gi,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        return (gi + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t gn,
                                    [[maybe_unused]] bool solver_full) const
        -> std::size_t override {
        return 2 * gn;
    }
};
}  // namespace poisson_solver::detail

#endif  // AXIS_STRATEGIES_HPP
