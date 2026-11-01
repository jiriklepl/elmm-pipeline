#include "poisson-solver/aliases.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/detail/helpers.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace poisson_solver::tests {
namespace psd = ::poisson_solver::detail;

template <std::floating_point T>
using RHSAdjustFunction = std::function<void(std::span<T>)>;

// === Forward Declarations ===
template <std::floating_point T>
auto MakeNopCorrector() -> RHSAdjustFunction<T>;

template <std::size_t Dim>
auto FlatToMultidimIdx(std::size_t idx,
                       GridSizes<Dim> ns) -> psd::MultidimIdx<Dim>;

// === Structs and Classes ===

/**
 * struct SolverConfig - Data for solver construction.
 * Exists mainly to reduce the amount of function params.
 */
template <std::floating_point T, std::size_t Dim>
struct SolverConfig {
public:
    GridSizes<Dim> ns{};
    std::array<T, Dim> lens{};
    BCs<Dim> bcs{};
    GridSizes<Dim> ngs_phi{};
    GridSizes<Dim> ngs_rhs{};
};

template <std::floating_point T, std::size_t Dim>
struct DistributedSolverConfig : public SolverConfig<T, Dim> {
    public:
        GridSizes<Dim> gns{};
        GridOffsets<Dim> offs{};
};

/**
 * struct NonuniformZSolverConfig - Data for nonuniform Z solver construction.
 * Exists mainly to reduce the amount of function params.
 */
template <std::floating_point T>
struct NonuniformZSolverConfig : SolverConfig<T, 3> {
public:
    std::vector<T> z{};
    std::vector<T> z_u{};
};

template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class MaxResiduumCalculator {
public:
    auto operator()(std::span<const T> phi,
                    std::span<const T> rhs,
                    GridSizes<Dim> ns,
                    BCs<Dim> bcs,
                    std::array<T, 2 * Dim> coeffs) const -> T {
        using psd::FlatIndex;

        assert(rhs.size() == phi.size());

        const std::size_t size = rhs.size();
        T max_residuum = 0.0;

        for (std::size_t i = 0; i < size; ++i) {
            T neighbor_sum = 0.0;
            T coeff_center = 0.0;
            auto midx = FlatToMultidimIdx<Dim>(i, ns);

            for (std::size_t d = 0; d < Dim; ++d) {
                const T coeff_left = coeffs[2 * d];
                const T coeff_right = coeffs[2 * d + 1];

                if (midx[d] > 0) {
                    auto left_idx = midx;
                    left_idx[d] -= 1;
                    neighbor_sum += phi[FlatIndex(left_idx, ns)] * coeff_left;
                    coeff_center += coeff_left;
                } else {
                    auto contrib = GetBorderContrib(
                        phi, ns, midx, d, Side::Left, bcs[d].first, coeff_left);
                    neighbor_sum += contrib.neighbor_sum;
                    coeff_center += contrib.coeff_center;
                }

                if (midx[d] < ns[d] - 1) {
                    auto right_idx = midx;
                    right_idx[d] += 1;
                    neighbor_sum += phi[FlatIndex(right_idx, ns)] * coeff_right;
                    coeff_center += coeff_right;
                } else {
                    auto contrib = GetBorderContrib(phi,
                                                    ns,
                                                    midx,
                                                    d,
                                                    Side::Right,
                                                    bcs[d].second,
                                                    coeff_right);
                    neighbor_sum += contrib.neighbor_sum;
                    coeff_center += contrib.coeff_center;
                }
            }

            neighbor_sum -= rhs[i];
            const T local_residuum =
                std::abs(coeff_center * phi[i] - neighbor_sum);
            max_residuum = std::max(max_residuum, local_residuum);
        }
        return max_residuum;
    }

protected:
    using MultidimIdx = psd::MultidimIdx<Dim>;

    enum class Side { Left, Right };

    struct BoundaryContribution {
        T neighbor_sum{};
        T coeff_center{};
    };

    auto GetBorderContrib(std::span<const T> phi,
                          GridSizes<Dim> ns,
                          MultidimIdx midx,
                          std::size_t dim,
                          Side side,
                          BC bc,
                          T coeff) const -> BoundaryContribution {
        using psd::FlatIndex;

        switch (bc) {
            case BC::Periodic: {
                auto wrap_idx = midx;
                wrap_idx[dim] = (side == Side::Left) ? (ns[dim] - 1) : 0;
                return {phi[FlatIndex(wrap_idx, ns)] * coeff, coeff};
            }
            case BC::Dirichlet: {
                return {0.0, coeff};
            }
            case BC::Neumann: {
                auto inner_idx = midx;
                inner_idx[dim] = (side == Side::Left) ? 1 : (ns[dim] - 2);
                return {phi[FlatIndex(inner_idx, ns)] * coeff, coeff};
            }
            case BC::DirichletStag: {
                return {-phi[FlatIndex(midx, ns)] * coeff, coeff};
            }
            case BC::NeumannStag: {
                return {0.0, 0.0};
            }
            default: {
                assert(false && "Unknown boundary condition.");
            }
        }

        // To prevent UB in release builds.
        throw std::runtime_error("We should not get here!");
    }
};

/**
 * struct TestConfig - Data specific to a single test-case.
 *
 * @tparam T Floating point type.
 * @tparam Dim Dimension of the tested solver.
 */
template <std::floating_point T, std::size_t Dim>
struct TestConfig {
public:
    TestConfig(BCs<Dim> bcs,
               std::array<std::size_t, Dim> dx_divisors,
               RHSAdjustFunction<T> rhs_adjust = MakeNopCorrector<T>())
        : bcs(bcs),
          dx_divisors(dx_divisors),
          rhs_adjust(rhs_adjust) {}

    BCs<Dim> bcs{};
    std::array<std::size_t, Dim> dx_divisors{};
    RHSAdjustFunction<T> rhs_adjust = nullptr;

    auto Name() const -> std::string {
        std::string name;
        for (std::size_t i{0}; const auto &bcp : bcs) {
            name +=
                psd::ToString(bcp.first) + " | " + psd::ToString(bcp.second);
            if (i < Dim - 1) {
                name += " | ";
            }
            ++i;
        }
        return name;
    }
};

template <std::floating_point T>
struct ResiduumResult {
public:
    bool success;
    T max_residuum;
};

// === Functions ===
template <std::floating_point T>
void Initialize(std::span<T> phi, std::span<T> rhs, unsigned seed) {
    assert(rhs.size() == phi.size());

    const std::size_t size = rhs.size();
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<T> partial_sums(num_threads, 0.0);

    auto rng_fill_worker = [&](unsigned int t_id) {
        std::size_t chunk = (size + num_threads - 1) / num_threads;
        std::size_t start = t_id * chunk;
        std::size_t end = std::min(start + chunk, size);

        std::mt19937 rng(seed + t_id);
        std::uniform_real_distribution<T> dist(0.0, 1.0);

        T local_sum = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            phi[i] = dist(rng);
            T rhs_val = dist(rng);
            rhs[i] = rhs_val;
            local_sum += rhs_val;
        }
        partial_sums[t_id] = local_sum;
    };

    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(rng_fill_worker, i);
    }
    for (auto &t : threads) {
        t.join();
    }
    threads.clear();

    T rhs_sum = 0.0;
    for (T s : partial_sums) {
        rhs_sum += s;
    }
    T avg = rhs_sum / static_cast<T>(size);

    auto avg_sub_worker = [&](unsigned int t_id) {
        std::size_t chunk = (size + num_threads - 1) / num_threads;
        std::size_t start = t_id * chunk;
        std::size_t end = std::min(start + chunk, size);

        for (std::size_t i = start; i < end; ++i) {
            rhs[i] -= avg;
        }
    };

    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(avg_sub_worker, i);
    }
    for (auto &t : threads) {
        t.join();
    }
}

template <std::floating_point T>
void InitializeZs(std::size_t nz,
                  T len_z,
                  std::vector<T> &z,
                  std::vector<T> &z_u,
                  unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-0.25),
                                           static_cast<T>(0.25));

    z_u.resize(nz + 1);
    for (std::size_t i = 0; i <= nz; ++i) {
        z_u[i] = (len_z / static_cast<T>(nz)) * static_cast<T>(i);
    }

    for (std::size_t i = 1; i < nz; ++i) {
        T p = dist(rng);
        T min_dist = std::min(std::abs(z_u[i + 1] - z_u[i]),
                              std::abs(z_u[i] - z_u[i - 1]));
        z_u[i] += min_dist * p;
    }

    T z_u_ghost_bottom = z_u[0] - (z_u[1] - z_u[0]);
    T z_u_ghost_top = z_u[nz] + (z_u[nz] - z_u[nz - 1]);

    z.resize(nz + 2);
    z[0] = (z_u[0] + z_u_ghost_bottom) / static_cast<T>(2.0);
    for (std::size_t i = 1; i <= nz; ++i) {
        z[i] = (z_u[i] + z_u[i - 1]) / static_cast<T>(2.0);
    }
    z[nz + 1] = (z_u_ghost_top + z_u[nz]) / static_cast<T>(2.0);
}

inline auto ToCBoundary(BC cxx_bc) -> ::BoundaryCondition {
    switch (cxx_bc) {
        case BC::Periodic:
            return Periodic;
        case BC::Dirichlet:
            return Dirichlet;
        case BC::Neumann:
            return Neumann;
        case BC::DirichletStag:
            return DirichletStag;
        case BC::NeumannStag:
            return NeumannStag;
        default:
            assert(false && "Unknown boundary condition.");
    }

    // To prevent UB in release builds.
    throw std::runtime_error("We should never get here!");
}

template <std::size_t Dim>
auto FlatToMultidimIdx(std::size_t idx,
                       GridSizes<Dim> ns) -> psd::MultidimIdx<Dim> {
    psd::MultidimIdx<Dim> midx{};
    for (std::size_t i = 0; i < Dim; ++i) {
        midx[i] = idx % ns[i];
        idx /= ns[i];
    }
    return midx;
};

template <std::floating_point T, std::size_t Dim>
auto ResiduumBelowThreshold(T residuum,
                            GridSizes<Dim> ns) -> ResiduumResult<T> {
    constexpr T fudge_factor = static_cast<T>(100.0);
    T max_residuum =
        std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>()) *
        fudge_factor * std::numeric_limits<T>::epsilon();
    bool success = residuum < max_residuum;
    return {.success = success, .max_residuum = max_residuum};
}

inline auto ParseBC(const std::string &str) -> BC {
    if (str == "P")
        return BC::Periodic;
    if (str == "D")
        return BC::Dirichlet;
    if (str == "N")
        return BC::Neumann;
    if (str == "DS")
        return BC::DirichletStag;
    if (str == "NS")
        return BC::NeumannStag;
    throw std::runtime_error("Unknown BC: " + str);
}

// == Correctors ==
/**
 * @brief Makes a corrector that does no-op.
 */
template <std::floating_point T>
auto MakeNopCorrector() -> RHSAdjustFunction<T> {
    return [](std::span<T>) {};
}

template <std::floating_point T>
auto MakePPDRhsCorrector() -> RHSAdjustFunction<T> {
    using std::ranges::for_each;
    return [](std::span<T> rhs) {
        for_each(rhs, [](T &val) {
            val += static_cast<T>(1.0);
        });
    };
}

template <std::floating_point T, std::size_t Dim>
auto MakeNeumannRhsCorrector(GridSizes<Dim> ns) -> RHSAdjustFunction<T> {
    return [ns](std::span<T> rhs) {
        T sum = 0.0;
        std::size_t num_cells = 1;

        // Calculate the total number of "cells" (N-1 per dimension)
        for (std::size_t d = 0; d < Dim; ++d) {
            num_cells *= ns[d] - 1;
        }

        for (std::size_t i = 0; i < rhs.size(); ++i) {
            T p = rhs[i];

            auto midx = FlatToMultidimIdx<Dim>(i, ns);

            // Halve the value for every dimension's boundary it touches
            for (std::size_t d = 0; d < Dim; ++d) {
                if (midx[d] == 0 || midx[d] == ns[d] - 1) {
                    p /= static_cast<T>(2.0);
                }
            }

            sum += p;
        }

        T mean = sum / num_cells;

        // Subtract the calculated mean from every node
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            rhs[i] -= mean;
        }
    };
}

template <std::floating_point T>
auto MakeNeumannStagRhsCorrector() -> RHSAdjustFunction<T> {
    return [](std::span<T> rhs) {
        T sum = std::accumulate(rhs.begin(), rhs.end(), 0.0);
        std::ranges::for_each(rhs, [sum](T &elem) {
            elem -= sum;
        });
    };
}

}  // namespace poisson_solver::tests
