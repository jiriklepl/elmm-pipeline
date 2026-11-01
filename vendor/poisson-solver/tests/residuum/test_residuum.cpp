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
#include <doctest.h>
#include <functional>
#include <numbers>
#include <random>
#include <vector>

namespace {
namespace ps = ::poisson_solver;
namespace pst = ps::tests;

using ps::BC;
using ps::BCs;
using ps::GridSizes;
using ps::IApproximation;
using pst::MaxResiduumCalculator;
using pst::SolverConfig;
using pst::TestConfig;

constexpr BC P = BC::Periodic;
constexpr BC D = BC::Dirichlet;
constexpr BC N = BC::Neumann;
constexpr BC DS = BC::DirichletStag;
constexpr BC NS = BC::NeumannStag;

template <std::floating_point T, std::size_t Dim>
void ExecuteCBinding(std::span<T> phi,
                     std::span<const T> rhs,
                     SolverConfig<T, Dim> cfg,
                     Approximation approx) {
    using pst::ToCBoundary;

    PoissonSolver solver;
    std::array<int, Dim> c_ns;
    std::array<T, Dim> c_lens;
    std::array<BoundaryCondition, 2 * Dim> c_bcs;
    std::array<int, Dim> c_ngs_phi;
    std::array<int, Dim> c_ngs_rhs;

    for (std::size_t i = 0; i < Dim; ++i) {
        c_ns[i] = static_cast<int>(cfg.ns[i]);
        c_lens[i] = cfg.lens[i];
        c_bcs[2 * i] = ToCBoundary(cfg.bcs[i].first);
        c_bcs[2 * i + 1] = ToCBoundary(cfg.bcs[i].second);
        c_ngs_phi[i] = static_cast<int>(cfg.ngs_phi[i]);
        c_ngs_rhs[i] = static_cast<int>(cfg.ngs_rhs[i]);
    }

    if constexpr (std::is_same_v<T, double>) {
        PoissonSolverNew(&solver,
                         Dim,
                         c_ns.data(),
                         c_lens.data(),
                         c_bcs.data(),
                         approx,
                         nullptr,
                         nullptr,
                         nullptr);
        PoissonSolverExecute(&solver,
                             phi.data(),
                             rhs.data(),
                             c_ngs_phi.data(),
                             c_ngs_rhs.data());
        PoissonSolverFinalize(&solver);
    } else {
        PoissonSolverNewFloat(&solver,
                              Dim,
                              c_ns.data(),
                              c_lens.data(),
                              c_bcs.data(),
                              approx,
                              nullptr,
                              nullptr,
                              nullptr);
        PoissonSolverExecuteFloat(&solver,
                                  phi.data(),
                                  rhs.data(),
                                  c_ngs_phi.data(),
                                  c_ngs_rhs.data());
        PoissonSolverFinalizeFloat(&solver);
    }
}

template <std::floating_point T, std::size_t Dim>
void ExecuteCBindingGPU(std::span<T> phi,
                        std::span<const T> rhs,
                        SolverConfig<T, Dim> cfg,
                        Approximation approx) {
    using pst::ToCBoundary;

    PoissonSolver solver;
    std::array<int, Dim> c_ns;
    std::array<T, Dim> c_lens;
    std::array<BoundaryCondition, 2 * Dim> c_bcs;

    std::array<int, Dim> c_ngs_phi;
    std::array<int, Dim> c_ngs_rhs;

    for (std::size_t i = 0; i < Dim; ++i) {
        c_ns[i] = static_cast<int>(cfg.ns[i]);
        c_lens[i] = cfg.lens[i];
        c_bcs[2 * i] = ToCBoundary(cfg.bcs[i].first);
        c_bcs[2 * i + 1] = ToCBoundary(cfg.bcs[i].second);
        c_ngs_phi[i] = static_cast<int>(cfg.ngs_phi[i]);
        c_ngs_rhs[i] = static_cast<int>(cfg.ngs_rhs[i]);
    }

    T *d_phi = nullptr;
    T *d_rhs = nullptr;
    std::size_t bytes = phi.size() * sizeof(T);
    cudaMalloc(&d_phi, bytes);
    cudaMalloc(&d_rhs, bytes);
    cudaMemcpy(d_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice);

    if constexpr (std::is_same_v<T, double>) {
        PoissonSolverNew(&solver,
                         Dim,
                         c_ns.data(),
                         c_lens.data(),
                         c_bcs.data(),
                         approx,
                         nullptr,
                         nullptr,
                         nullptr);
        PoissonSolverExecuteGPU(&solver, d_phi, d_rhs, c_ngs_phi.data(), c_ngs_rhs.data());
        PoissonSolverFinalize(&solver);
    } else {
        PoissonSolverNewFloat(&solver,
                              Dim,
                              c_ns.data(),
                              c_lens.data(),
                              c_bcs.data(),
                              approx,
                              nullptr,
                              nullptr,
                              nullptr);
        PoissonSolverExecuteGPUFloat(&solver, d_phi, d_rhs, c_ngs_phi.data(), c_ngs_rhs.data());
        PoissonSolverFinalizeFloat(&solver);
    }

    cudaMemcpy(phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_phi);
    cudaFree(d_rhs);
}

// Runs tests for both the C++ and C APIs
template <std::floating_point T, std::size_t Dim>
void RunApis(std::span<T> phi,
             std::span<const T> rhs,
             SolverConfig<T, Dim> cfg,
             const IApproximation<T> &approx,
             Approximation c_approx,
             const std::function<T()> &calc_residuum) {
    using ps::Solver;
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

    SUBCASE("API: C++") {
        Solver<T, Dim> solver(cfg.ns, cfg.lens, cfg.bcs, approx);
        solver.Execute(phi, rhs, cfg.ngs_phi, cfg.ngs_rhs);
        verify_residuum();
    }

    SUBCASE("API: C++ GPU") {
        Solver<T, Dim> solver(cfg.ns, cfg.lens, cfg.bcs, approx);

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
        ExecuteCBinding(phi, rhs, cfg, c_approx);
        verify_residuum();
    }

    SUBCASE("API: C Binding GPU") {
        ExecuteCBindingGPU(phi, rhs, cfg, c_approx);
        verify_residuum();
    }
}

template <std::floating_point T, std::size_t Dim>
void RunTestConfigs(std::span<const TestConfig<T, Dim>> configs,
                    std::span<T> phi,
                    std::span<const T> rhs,
                    GridSizes<Dim> ns,
                    std::array<T, Dim> lens) {
    for (const auto &tc : configs) {
        SUBCASE("Boundary: " + tc.Name()) {
            std::vector<T> local_rhs(rhs.begin(), rhs.end());
            tc.rhs_adjust(local_rhs);

            // Spectral approx tested only in Fortran tests

            SUBCASE("Approximation: FD2") {
                using ps::FiniteDifference2Approximation;

                std::array<T, 2 * Dim> coeffs{};
                for (std::size_t d = 0; d < Dim; ++d) {
                    T dx = lens[d] / tc.dx_divisors[d];
                    T inv_dx2 = static_cast<T>(1.0) / (dx * dx);

                    coeffs[2 * d] = inv_dx2;
                    coeffs[2 * d + 1] = inv_dx2;
                }

                MaxResiduumCalculator<T, Dim> mrc{};
                auto calc_max_residuum =
                    [&mrc, &phi, &local_rhs, &ns, &bcs = tc.bcs, &coeffs]() {
                        return mrc(phi, local_rhs, ns, bcs, coeffs);
                    };
                SolverConfig<T, Dim> cfg{ns, lens, tc.bcs};

                RunApis<T, Dim>(phi,
                                local_rhs,
                                cfg,
                                FiniteDifference2Approximation<T>{},
                                FD2,
                                calc_max_residuum);
            }
        }
    }
}

TEST_CASE_TEMPLATE("Poisson Solver Residuum Suite", T, float, double) {
    using pst::MakeNeumannRhsCorrector;
    using pst::MakeNeumannStagRhsCorrector;
    using pst::MakeNopCorrector;
    using pst::MakePPDRhsCorrector;

    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    // Values taken from `PoisFFT` tests
    constexpr std::size_t nx = 21;
    constexpr std::size_t ny = 32;
    constexpr std::size_t nz = 25;
    constexpr T lenx = 2 * std::numbers::pi_v<T>;
    constexpr T leny = lenx * 1.1;
    constexpr T lenz = lenx / 1.1;

    SUBCASE("1D") {
        using pst::Initialize;
        using pst::TestConfig;

        constexpr GridSizes<1> ns1d{nx};
        constexpr std::array<T, 1> lens1d = {lenx};
        std::vector<T> phi1d(nx);
        std::vector<T> rhs1d(nx);

        std::vector<TestConfig<T, 1>> configs = {
            {{{{P, P}}}, {nx}},
            {{{{D, D}}}, {(nx + 1)}},
            {{{{N, N}}}, {(nx - 1)}, MakeNeumannRhsCorrector<T, 1>(ns1d)},
            {{{{DS, DS}}}, {nx}},
            {{{{NS, NS}}}, {nx}, MakeNeumannStagRhsCorrector<T>()},
            {{{{NS, DS}}}, {nx}},
            {{{{DS, NS}}}, {nx}},
            {{{{D, N}}}, {nx}},
            {{{{N, D}}}, {nx}}};

        Initialize<T>(phi1d, rhs1d, seed);
        RunTestConfigs<T, 1>(configs, phi1d, rhs1d, ns1d, lens1d);
    }

    SUBCASE("2D") {
        using pst::Initialize;
        using pst::TestConfig;

        constexpr GridSizes<2> ns2d{nx, ny};
        constexpr std::array<T, 2> lens2d = {lenx, leny};
        std::vector<T> phi2d(nx * ny);
        std::vector<T> rhs2d(nx * ny);

        std::vector<TestConfig<T, 2>> configs = {
            {{{{P, P}, {P, P}}}, {nx, ny}},
            {{{{D, D}, {D, D}}}, {(nx + 1), (ny + 1)}},
            {{{{N, N}, {N, N}}},
             {(nx - 1), (ny - 1)},
             MakeNeumannRhsCorrector<T, 2>(ns2d)},
            {{{{DS, DS}, {DS, DS}}}, {nx, ny}},
            {{{{NS, NS}, {NS, NS}}},
             {nx, ny},
             MakeNeumannStagRhsCorrector<T>()}};

        Initialize<T>(phi2d, rhs2d, seed);
        RunTestConfigs<T, 2>(configs, phi2d, rhs2d, ns2d, lens2d);
    }

    SUBCASE("3D") {
        using pst::Initialize;
        using pst::TestConfig;

        constexpr GridSizes<3> ns3d{nx, ny, nz};
        constexpr std::array<T, 3> lens3d = {lenx, leny, lenz};
        std::vector<T> phi3d(nx * ny * nz);
        std::vector<T> rhs3d(nx * ny * nz);

        std::vector<TestConfig<T, 3>> configs = {
            {{{{P, P}, {P, P}, {P, P}}}, {nx, ny, nz}},
            {{{{D, D}, {D, D}, {D, D}}}, {(nx + 1), (ny + 1), (nz + 1)}},
            {{{{N, N}, {N, N}, {N, N}}},
             {(nx - 1), (ny - 1), (nz - 1)},
             MakeNeumannRhsCorrector<T, 3>(ns3d)},
            {{{{DS, DS}, {DS, DS}, {DS, DS}}}, {nx, ny, nz}},
            {{{{NS, NS}, {NS, NS}, {NS, NS}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {NS, NS}}}, {{nx, ny, nz}}},
            {{{{P, P}, {P, P}, {DS, DS}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {DS, NS}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {NS, DS}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {D, N}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {N, D}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{NS, NS}, {P, P}, {P, P}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{P, P}, {NS, NS}, {P, P}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{P, P}, {NS, NS}, {NS, NS}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{DS, DS}, {DS, DS}, {P, P}}}, {nx, ny, nz}},
            {{{{DS, DS}, {DS, DS}, {NS, NS}}}, {nx, ny, nz}},
            {{{{DS, DS}, {DS, DS}, {NS, DS}}}, {nx, ny, nz}},
            {{{{D, D}, {D, D}, {N, D}}}, {nx + 1, ny + 1, nz}},
            {{{{NS, DS}, {NS, NS}, {NS, NS}}}, {nx, ny, nz}},
            {{{{N, D}, {N, N}, {N, N}}}, {nx, ny - 1, nz - 1}}};

        Initialize<T>(phi3d, rhs3d, seed);
        RunTestConfigs<T, 3>(configs, phi3d, rhs3d, ns3d, lens3d);
    }
}
}  // namespace
