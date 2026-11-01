#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/solver.hpp"
#include "test_helpers.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <doctest.h>
#include <random>
#include <vector>

namespace {
namespace ps = ::poisson_solver;
namespace psd = ps::detail;
namespace pst = ps::tests;

using ps::BC;
using ps::BCs;
using ps::GridSizes;
using ps::IApproximation;
using pst::MaxResiduumCalculator;
using pst::NonuniformZSolverConfig;
using pst::TestConfig;

constexpr BC P = BC::Periodic;
constexpr BC NS = BC::NeumannStag;



template <std::floating_point T>
void InitializeNonuniformZ(std::span<T> phi,
                           std::span<T> rhs,
                           GridSizes<3> ns,
                           T dx,
                           T dy,
                           const std::vector<T> &z_u,
                           std::array<T, 3> lens,
                           unsigned seed) {
    assert(rhs.size() == phi.size());

    using psd::MultidimIdx;
    using pst::FlatToMultidimIdx;
    using std::ranges::for_each;

    const std::size_t size = rhs.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    T avg = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        auto midx = FlatToMultidimIdx(i, ns);
        auto k = midx[2];
        auto dz_u = z_u[k + 1] - z_u[k];
        phi[i] = dist(rng);
        rhs[i] = dist(rng);
        avg += rhs[i] * dx * dy * dz_u;
    }

    avg /= (lens[0] * lens[1] * lens[2]);

    for_each(rhs, [avg](auto &&elem) {
        elem -= avg;
    });
}

template <std::floating_point T>
class MaxResiduumCalculatorNonuniformZ : MaxResiduumCalculator<T, 3> {
public:
    auto operator()(std::span<const T> phi,
                    std::span<const T> rhs,
                    GridSizes<3> ns,
                    BCs<3> bcs,
                    const std::vector<T> &z,
                    const std::vector<T> &z_u,
                    std::array<T, 4> xy_coeffs) const -> T {
        using ps::detail::FlatIndex;
        using pst::FlatToMultidimIdx;

        assert(rhs.size() == phi.size());
        T max_residuum = 0.0;

        for (std::size_t i = 0; i < rhs.size(); ++i) {
            T neighbor_sum = 0.0;
            T coeff_center = 0.0;
            auto midx = FlatToMultidimIdx<3>(i, ns);

            // Compute XY dimensions just like for uniform solver
            for (std::size_t d = 0; d < 2; ++d) {
                const T coeff_left = xy_coeffs[2 * d];
                const T coeff_right = xy_coeffs[2 * d + 1];

                if (midx[d] > 0) {
                    auto left_idx = midx;
                    left_idx[d] -= 1;
                    neighbor_sum += phi[FlatIndex(left_idx, ns)] * coeff_left;
                    coeff_center += coeff_left;
                } else {
                    auto contrib = this->GetBorderContrib(
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
                    auto contrib = this->GetBorderContrib(phi,
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

            // Compute Z Dimension (Variable Coefficients)
            std::size_t k = midx[2];

            // Distances mapping:
            // centers[k+1] is the center of the current cell k.
            // centers[k] is the center of the cell below.
            // centers[k+2] is the center of the cell above.
            T dz_c_left = z[k + 1] - z[k];
            T dz_c_right = z[k + 2] - z[k + 1];
            T dz_u = z_u[k + 1] - z_u[k];

            T coeff_left_z = static_cast<T>(1.0) / (dz_c_left * dz_u);
            T coeff_right_z = static_cast<T>(1.0) / (dz_c_right * dz_u);

            if (k > 0) {
                auto bot_idx = midx;
                bot_idx[2] -= 1;
                neighbor_sum += phi[FlatIndex(bot_idx, ns)] * coeff_left_z;
                coeff_center += coeff_left_z;
            } else {
                auto contrib = this->GetBorderContrib(
                    phi, ns, midx, 2, Side::Left, bcs[2].first, coeff_left_z);
                neighbor_sum += contrib.neighbor_sum;
                coeff_center += contrib.coeff_center;
            }

            if (k < ns[2] - 1) {
                auto top_idx = midx;
                top_idx[2] += 1;
                neighbor_sum += phi[FlatIndex(top_idx, ns)] * coeff_right_z;
                coeff_center += coeff_right_z;
            } else {
                auto contrib = this->GetBorderContrib(phi,
                                                      ns,
                                                      midx,
                                                      2,
                                                      Side::Right,
                                                      bcs[2].second,
                                                      coeff_right_z);
                neighbor_sum += contrib.neighbor_sum;
                coeff_center += contrib.coeff_center;
            }

            neighbor_sum -= rhs[i];
            const T local_residuum =
                std::abs(coeff_center * phi[i] - neighbor_sum);
            max_residuum = std::max(max_residuum, local_residuum);
        }

        return max_residuum;
    }

private:
    using Side = MaxResiduumCalculator<T, 3>::Side;
};

template <std::floating_point T>
void ExecuteCBindingNonuniformZ(std::span<T> phi,
                                std::span<const T> rhs,
                                const NonuniformZSolverConfig<T> &cfg,
                                Approximation approx) {
    using pst::ToCBoundary;

    PoissonSolver solver;
    std::array<int, 3> c_ns;
    std::array<int, 3> c_ngs_phi;
    std::array<int, 3> c_ngs_rhs;
    std::array<T, 2> c_lens_xy;
    std::array<BoundaryCondition, 6> c_bcs;

    for (std::size_t i = 0; i < 3; ++i) {
        c_ns[i] = static_cast<int>(cfg.ns[i]);
        c_bcs[2 * i] = ToCBoundary(cfg.bcs[i].first);
        c_bcs[2 * i + 1] = ToCBoundary(cfg.bcs[i].second);
        c_ngs_phi[i] = static_cast<int>(cfg.ngs_phi[i]);
        c_ngs_rhs[i] = static_cast<int>(cfg.ngs_rhs[i]);
    }

    c_lens_xy[0] = cfg.lens[0];
    c_lens_xy[1] = cfg.lens[1];

    if constexpr (std::is_same_v<T, double>) {
        PoissonSolverNonuniformZNew(&solver,
                                    c_ns.data(),
                                    c_lens_xy.data(),
                                    cfg.z.data(),
                                    cfg.z_u.data(),
                                    c_bcs.data(),
                                    approx,
                                    nullptr,
                                    nullptr,
                                    nullptr);
        PoissonSolverNonuniformZExecute(&solver,
                                        phi.data(),
                                        rhs.data(),
                                        c_ngs_phi.data(),
                                        c_ngs_rhs.data());
        PoissonSolverNonuniformZFinalize(&solver);
    } else {
        PoissonSolverNonuniformZNewFloat(&solver,
                                         c_ns.data(),
                                         c_lens_xy.data(),
                                         cfg.z.data(),
                                         cfg.z_u.data(),
                                         c_bcs.data(),
                                         approx,
                                         nullptr,
                                         nullptr,
                                         nullptr);
        PoissonSolverNonuniformZExecuteFloat(&solver,
                                             phi.data(),
                                             rhs.data(),
                                             c_ngs_phi.data(),
                                             c_ngs_rhs.data());
        PoissonSolverNonuniformZFinalizeFloat(&solver);
    }
}

template <std::floating_point T>
void ExecuteCBindingNonuniformZGPU(std::span<T> phi,
                                   std::span<const T> rhs,
                                   const NonuniformZSolverConfig<T> &cfg,
                                   Approximation approx) {
    using pst::ToCBoundary;

    PoissonSolver solver;
    std::array<int, 3> c_ns;
    std::array<T, 2> c_lens_xy;
    std::array<BoundaryCondition, 6> c_bcs;

    std::array<int, 3> c_ngs_phi;
    std::array<int, 3> c_ngs_rhs;

    for (std::size_t i = 0; i < 3; ++i) {
        c_ns[i] = static_cast<int>(cfg.ns[i]);
        c_bcs[2 * i] = ToCBoundary(cfg.bcs[i].first);
        c_bcs[2 * i + 1] = ToCBoundary(cfg.bcs[i].second);
        c_ngs_phi[i] = static_cast<int>(cfg.ngs_phi[i]);
        c_ngs_rhs[i] = static_cast<int>(cfg.ngs_rhs[i]);
    }

    c_lens_xy[0] = cfg.lens[0];
    c_lens_xy[1] = cfg.lens[1];

    T *d_phi = nullptr;
    T *d_rhs = nullptr;
    std::size_t bytes = phi.size() * sizeof(T);
    cudaMalloc(&d_phi, bytes);
    cudaMalloc(&d_rhs, bytes);
    cudaMemcpy(d_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice);

    if constexpr (std::is_same_v<T, double>) {
        PoissonSolverNonuniformZNew(&solver,
                                    c_ns.data(),
                                    c_lens_xy.data(),
                                    cfg.z.data(),
                                    cfg.z_u.data(),
                                    c_bcs.data(),
                                    approx,
                                    nullptr,
                                    nullptr,
                                    nullptr);
        PoissonSolverNonuniformZExecuteGPU(&solver, d_phi, d_rhs, c_ngs_phi.data(), c_ngs_rhs.data());
        PoissonSolverNonuniformZFinalize(&solver);
    } else {
        PoissonSolverNonuniformZNewFloat(&solver,
                                         c_ns.data(),
                                         c_lens_xy.data(),
                                         cfg.z.data(),
                                         cfg.z_u.data(),
                                         c_bcs.data(),
                                         approx,
                                         nullptr,
                                         nullptr,
                                         nullptr);
        PoissonSolverNonuniformZExecuteGPUFloat(&solver, d_phi, d_rhs, c_ngs_phi.data(), c_ngs_rhs.data());
        PoissonSolverNonuniformZFinalizeFloat(&solver);
    }

    cudaMemcpy(phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_phi);
    cudaFree(d_rhs);
}

template <std::floating_point T>
void RunApisNonuniformZ(std::span<T> phi,
                        std::span<const T> rhs,
                        const NonuniformZSolverConfig<T> &cfg,
                        const IApproximation<T> &approx,
                        Approximation c_approx,
                        const std::function<T()> &calc_residuum) {
    using ps::NonuniformZSolver;
    using pst::ResiduumBelowThreshold;

    auto verify_residuum = [&calc_residuum, ns = cfg.ns]() {
        T residuum = calc_residuum();
        CHECK_MESSAGE(residuum != T{0.0},
                      "Residuum is exactly 0.0 (suspicious).");

        auto result = ResiduumBelowThreshold<T>(residuum, ns);
        CHECK_MESSAGE(result.success,
                      "Residuum is ",
                      residuum,
                      " (Max allowed: ",
                      result.max_residuum,
                      ")");
    };

    auto lens_xy = std::array{cfg.lens[0], cfg.lens[1]};

    SUBCASE("API: C++") {
        NonuniformZSolver<T> solver{
            cfg.ns, lens_xy, cfg.z, cfg.z_u, cfg.bcs, approx};
        solver.Execute(phi, rhs, cfg.ngs_phi, cfg.ngs_rhs);
        verify_residuum();
    }

    SUBCASE("API: C++ GPU") {
        NonuniformZSolver<T> solver{
            cfg.ns, lens_xy, cfg.z, cfg.z_u, cfg.bcs, approx};

        T *d_phi = nullptr;
        T *d_rhs = nullptr;
        std::size_t bytes = phi.size() * sizeof(T);

        cudaMalloc(&d_phi, bytes);
        cudaMalloc(&d_rhs, bytes);

        cudaMemcpy(d_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice);

        solver.ExecuteGPU(std::span{d_phi, phi.size()},
                          std::span{d_rhs, rhs.size()},
                          cfg.ngs_phi,
                          cfg.ngs_rhs);

        cudaMemcpy(phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_phi);
        cudaFree(d_rhs);

        verify_residuum();
    }

    SUBCASE("API: C Binding") {
        ExecuteCBindingNonuniformZ(phi, rhs, cfg, c_approx);
        verify_residuum();
    }

    SUBCASE("API: C Binding GPU") {
        ExecuteCBindingNonuniformZGPU(phi, rhs, cfg, c_approx);
        verify_residuum();
    }
}

template <std::floating_point T>
void RunTestConfigsNonuniformZ(std::span<const TestConfig<T, 3>> configs,
                               std::span<T> phi,
                               std::span<const T> rhs,
                               GridSizes<3> ns,
                               std::array<T, 3> lens,
                               const std::vector<T> &z,
                               const std::vector<T> &z_u) {
    for (const auto &tc : configs) {
        SUBCASE("Boundary: " + tc.Name()) {
            std::vector<T> local_rhs(rhs.begin(), rhs.end());
            tc.rhs_adjust(local_rhs);

            SUBCASE("Approximation: FD2") {
                using ps::FiniteDifference2Approximation;

                std::array<T, 4> xy_coeffs{};
                for (std::size_t d = 0; d < 2; ++d) {
                    T dx = lens[d] / static_cast<T>(ns[d]);
                    T inv_dx2 = static_cast<T>(1.0) / (dx * dx);
                    xy_coeffs[2 * d] = inv_dx2;
                    xy_coeffs[2 * d + 1] = inv_dx2;
                }

                MaxResiduumCalculatorNonuniformZ<T> mrc{};
                auto calc_max_residuum = [&]() {
                    return mrc(phi, local_rhs, ns, tc.bcs, z, z_u, xy_coeffs);
                };

                // Only the internal part of `z` is passed to the solver
                std::vector<T> z_internal(z.begin() + 1, z.end() - 1);
                NonuniformZSolverConfig<T> cfg{
                    {ns, lens, tc.bcs}, z_internal, z_u};
                RunApisNonuniformZ<T>(phi,
                                      local_rhs,
                                      cfg,
                                      FiniteDifference2Approximation<T>{},
                                      FD2,
                                      calc_max_residuum);
            }
        }
    }
}

TEST_CASE_TEMPLATE("Poisson Solver Non-uniform Z Residuum Suite",
                   T,
                   float,
                   double) {
    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    constexpr std::size_t nx = 21;
    constexpr std::size_t ny = 32;
    constexpr std::size_t nz = 25;
    constexpr T lenx = 2 * std::numbers::pi_v<T>;
    constexpr T leny = lenx * 1.1;
    constexpr T lenz = lenx / 1.1;

    constexpr T dx = lenx / nx;
    constexpr T dy = leny / ny;

    SUBCASE("3D") {
        constexpr GridSizes<3> ns{nx, ny, nz};
        constexpr std::array<T, 3> lens = {lenx, leny, lenz};
        std::vector<T> phi(nx * ny * nz);
        std::vector<T> rhs(nx * ny * nz);
        std::vector<T> z_u;
        std::vector<T> z;

        pst::InitializeZs<T>(nz, lenz, z, z_u, seed);
        InitializeNonuniformZ<T>(phi, rhs, ns, dx, dy, z_u, lens, seed);

        std::vector<TestConfig<T, 3>> configs = {
            {{{{P, P}, {P, P}, {NS, NS}}}, {nx, ny, nz}},
            {{{{NS, NS}, {NS, NS}, {NS, NS}}}, {nx, ny, nz}}};

        RunTestConfigsNonuniformZ<T>(configs, phi, rhs, ns, lens, z, z_u);
    }
}
}  // namespace
