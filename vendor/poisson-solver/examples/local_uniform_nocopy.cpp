#include <array>
#include <cstddef>
#include <iostream>
#include <span>
#include <vector>

#include <cuda_runtime.h>

// You only need to include solver.hpp
#include <poisson-solver/solver.hpp>

using Prec = float;
using namespace poisson_solver;

int main() {
    // We will construct a 3D solver
    constexpr std::size_t dim = 3;

    // Define the grid side sizes
    constexpr std::size_t side_size = 64;
    GridSizes<dim> ns{side_size, side_size, side_size};

    // Define the grid side lengths
    std::array<Prec, dim> lens{1.0, 1.0, 1.0};

    // Define the boundary conditions
    // in the order of {west, east, south, north, bottom, top}
    constexpr auto Ns = BC::NeumannStag;
    BCs<dim> bcs{Ns, Ns, Ns, Ns, Ns, Ns};

    // Initialize the solver
    // The approximation is optional and SpectralApproximation as default
    Solver<Prec, dim> solver{
        ns, lens, bcs, FiniteDifference2Approximation<Prec>{}};

    // Grid will have two ghost cells on every side
    // One axis will therefore have four ghost cells
    // (two on the left, two on the right)
    GridSizes<dim> ngs{2, 2, 2};

    // The buffers hold the grid and the ghost cells on both sides
    std::size_t buffer_size{1};
    for (std::size_t d{0}; d < dim; ++d) {
        buffer_size *= ns[d] + 2 * ngs[d];
    }
    std::size_t bytes = buffer_size * sizeof(Prec);

    // Allocate the input and result buffers directly on the device.
    Prec *d_phi = nullptr;
    Prec *d_rhs = nullptr;
    cudaMalloc(&d_phi, bytes);
    cudaMalloc(&d_rhs, bytes);

    // A real application would already have the right-hand side on the
    // device (for example, produced there by a previous kernel). Here we
    // fill a host buffer instead and copy it over, purely to have some
    // input data for the example.
    std::vector<Prec> rhs(buffer_size, 1.0);
    cudaMemcpy(d_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice);

    // Solve the equation directly on the device buffers.
    // No host<->device copy of phi or rhs happens inside the solver call
    // itself.
    // (phi and rhs have the same amount of ghost cells)
    solver.ExecuteGPU(std::span<Prec>{d_phi, buffer_size},
                      std::span<const Prec>{d_rhs, buffer_size},
                      ngs, ngs);

    // Copy the result back to the host to inspect it.
    std::vector<Prec> phi(buffer_size, 0.0);
    cudaMemcpy(phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_phi);
    cudaFree(d_rhs);

    std::cout << "Successfully finished example local_uniform_nocopy\n";
}
