#include "poisson-solver/aliases.hpp"
#include "poisson-solver/solver.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poisson_solver;
using namespace tests;

template <std::floating_point T, std::size_t Dim>
auto InstantiateSolver(bool use_nonuniform_z,
                       tests::SolverConfig<T, Dim> &&cfg,
                       unsigned seed) -> std::unique_ptr<ISolver<T, Dim>> {
    std::unique_ptr<ISolver<T, Dim>> solver_ptr;
    if constexpr (Dim == 3) {
        if (use_nonuniform_z) {
            std::vector<T> z;
            std::vector<T> z_u;
            tests::InitializeZs<T>(cfg.ns[2], cfg.lens[2], z, z_u, seed);
            std::vector<T> z_internal(z.begin() + 1, z.end() - 1);
            std::array<T, 2> lens_xy{cfg.lens[0], cfg.lens[1]};
            solver_ptr = std::make_unique<NonuniformZSolver<T>>(
                cfg.ns, lens_xy, z_internal, z_u, cfg.bcs);
        } else {
            solver_ptr =
                std::make_unique<Solver<T, Dim>>(cfg.ns, cfg.lens, cfg.bcs);
        }
    } else {
        if (use_nonuniform_z) {
            throw std::runtime_error("nonuniform-z solver must have dim == 3");
        }
        solver_ptr =
            std::make_unique<Solver<T, Dim>>(cfg.ns, cfg.lens, cfg.bcs);
    }
    return solver_ptr;
}

template <std::floating_point T, std::size_t Dim>
void RunBenchmarks(std::size_t dim,
                   bool use_gpu,
                   bool use_nonuniform_z,
                   std::size_t measurement_iters,
                   std::size_t grid_size,
                   std::size_t grid_size_end,
                   std::size_t grid_size_step,
                   std::ofstream &out_file,
                   const std::vector<BC> &parsed_bcs,
                   unsigned seed) {
    constexpr T lenx = 2 * std::numbers::pi_v<T>;
    constexpr T leny = lenx * 1.1;
    constexpr T lenz = lenx / 1.1;
    std::array<T, Dim> lens;
    if constexpr (Dim >= 1)
        lens[0] = lenx;
    if constexpr (Dim >= 2)
        lens[1] = leny;
    if constexpr (Dim >= 3)
        lens[2] = lenz;

    BCs<Dim> bcs;
    for (std::size_t i = 0; i < Dim; ++i) {
        bcs[i].first = parsed_bcs[2 * i];
        bcs[i].second = parsed_bcs[2 * i + 1];
    }

    tests::TestConfig<T, Dim> cfg(bcs, {});
    std::string bc_name = cfg.Name();

    GridSizes<Dim> ngs{};
    ngs.fill(1);

    T *d_phi = nullptr;
    T *d_rhs = nullptr;

    if (use_gpu) {
        auto max_padded = grid_size_end + 2 * ngs[0];
        auto max_grid_size = max_padded * max_padded * max_padded;
        cudaMalloc(&d_phi, max_grid_size * sizeof(T));
        cudaMalloc(&d_rhs, max_grid_size * sizeof(T));
    }

    for (std::size_t size = grid_size; size <= grid_size_end;
         size += grid_size_step) {
        std::cout << "Measuring grid size " << size << std::endl;
        GridSizes<Dim> ns;
        std::size_t padded_elements = 1;
        for (std::size_t i = 0; i < Dim; ++i) {
            ns[i] = size;
            padded_elements *= size + 2 * ngs[i];
        }

        auto solver_ptr = InstantiateSolver<T, Dim>(
            use_nonuniform_z, {ns, lens, bcs, ngs, ngs}, seed);

        std::vector<T> h_phi(padded_elements);
        std::vector<T> h_rhs(padded_elements);

        for (std::size_t iter = 0; iter < measurement_iters; ++iter) {
            std::cout << "    Iteration " << iter << std::endl;
            tests::Initialize<T>(h_phi, h_rhs, seed + iter);

            ExecutionTimes times{};
            double outside_time_ms = 0.0;
            if (use_gpu) {
                cudaMemcpy(d_rhs,
                           h_rhs.data(),
                           padded_elements * sizeof(T),
                           cudaMemcpyHostToDevice);

                auto start = std::chrono::steady_clock::now();
                times = solver_ptr->ExecuteGPU(
                    std::span<T>{d_phi, padded_elements},
                    std::span<const T>{d_rhs, padded_elements},
                    ngs, ngs);
                auto end = std::chrono::steady_clock::now();
                outside_time_ms =
                    std::chrono::duration<double, std::milli>(end - start)
                        .count();
            } else {
                auto start = std::chrono::steady_clock::now();
                times = solver_ptr->Execute(h_phi, h_rhs, ngs, ngs);
                auto end = std::chrono::steady_clock::now();
                outside_time_ms =
                    std::chrono::duration<double, std::milli>(end - start)
                        .count();
            }

            std::string solver_type =
                use_nonuniform_z ? "nonuniform_z" : "uniform";
            out_file << Dim << "," << (sizeof(T) == 4 ? "float" : "double")
                     << "," << solver_type << "," << (use_gpu ? "GPU" : "CPU")
                     << ","
                     << "\"" << bc_name << "\"," << size << "," << iter << ","
                     << times.copy_time_ms << "," << times.reshape_time_ms
                     << "," << times.fft_time_ms << "," << times.kernel_time_ms
                     << "," << outside_time_ms << "," << times.SumTimes()
                     << std::endl;
        }
    }
    if (use_gpu) {
        cudaFree(d_phi);
        cudaFree(d_rhs);
    }
}

template <std::floating_point T, typename... Args>
void RunBenchmarks(std::size_t dim, Args &&...args) {
    switch (dim) {
        case 1:
            return RunBenchmarks<T, 1>(dim, std::forward<Args>(args)...);
        case 2:
            return RunBenchmarks<T, 2>(dim, std::forward<Args>(args)...);
        case 3:
            return RunBenchmarks<T, 3>(dim, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Dim out of bounds");
    }
}

auto main(int argc, char *argv[]) -> int {
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0]
                  << " float_precision solver_dim use_gpu use_nonuniform_z "
                     "measurement_iters "
                     "grid_side_size_begin grid_side_size_end "
                     "grid_side_size_step out_file BC_1 BC_2 ...\n";
        return 1;
    }
    std::string float_precision = argv[1];
    std::size_t dim = std::atoll(argv[2]);
    std::string use_gpu_str = argv[3];
    bool use_gpu =
        (use_gpu_str == "true" || use_gpu_str == "1" || use_gpu_str == "yes");
    std::string use_nonuniform_z_str = argv[4];
    bool use_nonuniform_z =
        (use_nonuniform_z_str == "true" || use_nonuniform_z_str == "1" ||
         use_nonuniform_z_str == "yes");
    std::size_t measurement_iters = std::atoll(argv[5]);
    std::size_t grid_size = std::atoll(argv[6]);
    std::size_t grid_size_end = std::atoll(argv[7]);
    std::size_t grid_size_step = std::atoll(argv[8]);
    std::string out_filename = argv[9];

    if (static_cast<std::size_t>(argc) != 10 + 2 * dim) {
        std::cerr << "Error: Expected " << 2 * dim
                  << " BC arguments for dimension " << dim << std::endl;
        return 1;
    }

    std::vector<BC> parsed_bcs;
    for (int i = 10; i < argc; ++i) {
        parsed_bcs.push_back(ParseBC(argv[i]));
    }

    std::ofstream out_file{out_filename, std::ios::trunc};
    if (!out_file.is_open()) {
        std::cerr << "Cannot open file \"" << out_filename << "\"" << std::endl;
        return 1;
    }

    out_file << "dim,precision,solver_type,api,bc_name,grid_size,iteration,copy_time_ms,"
                "reshape_time_ms,fft_time_ms,kernel_time_ms,outside_time_ms,"
                "sum_time_ms\n";

    std::random_device dev;
    unsigned seed = dev();

    if (float_precision == "float") {
        RunBenchmarks<float>(dim,
                             use_gpu,
                             use_nonuniform_z,
                             measurement_iters,
                             grid_size,
                             grid_size_end,
                             grid_size_step,
                             out_file,
                             parsed_bcs,
                             seed);
    } else if (float_precision == "double") {
        RunBenchmarks<double>(dim,
                              use_gpu,
                              use_nonuniform_z,
                              measurement_iters,
                              grid_size,
                              grid_size_end,
                              grid_size_step,
                              out_file,
                              parsed_bcs,
                              seed);
    } else {
        throw std::runtime_error("Precision is not 'float' or 'double'");
    }

    return 0;
}
