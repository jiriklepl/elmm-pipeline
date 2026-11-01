#ifndef APPROXIMATIONS_HPP
#define APPROXIMATIONS_HPP

#include <cmath>
#include <concepts>

/**
 * @file
 * @brief Contains all the approximation types supported by PoisFFT.
 */

namespace poisson_solver {
template <std::floating_point T>
class IApproximation {
public:
    virtual ~IApproximation() = default;
    virtual auto operator()(T x) const -> T = 0;
};

template <std::floating_point T>
class SpectralApproximation : public IApproximation<T> {
public:
    // f(x) = -(2x)^2
    auto operator()(T x) const -> T override {
        return static_cast<T>(-4.0) * x * x;
    }
};

template <std::floating_point T>
class FiniteDifference2Approximation : public IApproximation<T> {
public:
    // f(x) = -(2 * sin(x))^2
    auto operator()(T x) const -> T override {
        return static_cast<T>(-4.0) * std::sin(x) * std::sin(x);
    }
};

template <std::floating_point T>
class FiniteDifference4Approximation : public IApproximation<T> {
    // f(x) = -((sin(3x) - 27*sin(x)) / 12)^2
    auto operator()(T x) const -> T override {
        const T expr = (std::sin(3 * x) - 27 * std::sin(x)) / 12;
        return -(expr * expr);
    }
};
}  // namespace poisson_solver

#endif  // APPROXIMATIONS_HPP
