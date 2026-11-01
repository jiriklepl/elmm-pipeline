#include "poisson-solver/c_binding.h"

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/mpi_utils.hpp"
#include "poisson-solver/solver.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#ifdef POISSON_SOLVER_USE_MPI
#include <mpi.h>
#endif  // POISSON_SOLVER_USE_MPI

/**
 * @file
 * @brief Maps the C API to C++ API.
 */

namespace ps = poisson_solver;

namespace {
/**
 * @brief Map C approximation to C++ approximation.
 */
template <std::floating_point T>
auto GetApprox(Approximation approx) -> std::unique_ptr<ps::IApproximation<T>> {
    switch (approx) {
        case Spectral:
            return std::make_unique<ps::SpectralApproximation<T>>();
        case FD2:
            return std::make_unique<ps::FiniteDifference2Approximation<T>>();
        case FD4:
            return std::make_unique<ps::FiniteDifference4Approximation<T>>();
        default:
            throw std::invalid_argument(
                "Invalid approximation: " + std::to_string(approx) + ".");
    }
}

/**
 * @brief Map array of ints into GridSizes<Dim>
 *
 * @tparam Dim Array size.
 * @param ns The array.
 */
template <std::size_t Dim>
auto ToGridSizes(const int *ns) -> ps::GridSizes<Dim> {
    ps::GridSizes<Dim> gs{};
    std::transform(ns, ns + Dim, gs.begin(), [](int n) {
        return static_cast<std::size_t>(n);
    });
    return gs;
}

/**
 * @brief Map C boundary condition into C++ boundary condition.
 */
auto ToCxxBC(BoundaryCondition c_bc) -> ps::BoundaryCondition {
    using CxxBC = ps::BoundaryCondition;

    constexpr auto cxx_p = CxxBC::Periodic;
    constexpr auto cxx_d = CxxBC::Dirichlet;
    constexpr auto cxx_n = CxxBC::Neumann;
    constexpr auto cxx_ds = CxxBC::DirichletStag;
    constexpr auto cxx_ns = CxxBC::NeumannStag;

    switch (c_bc) {
        case Periodic:
            return cxx_p;
        case Dirichlet:
            return cxx_d;
        case Neumann:
            return cxx_n;
        case DirichletStag:
            return cxx_ds;
        case NeumannStag:
            return cxx_ns;
        default:
            throw std::invalid_argument(
                "Invalid boundary condition: " + std::to_string(c_bc) + ".");
    }
}

/**
 * @brief Map two C boundary conditions into a C++ BCPair.
 */
auto ToCxxBCPair(BoundaryCondition c_bc1, BoundaryCondition c_bc2)
    -> ps::BCPair {
    return ps::BCPair{ToCxxBC(c_bc1), ToCxxBC(c_bc2)};
}
}  // namespace

#define DEFINE_EXECUTE(DIM)                                          \
    void PoissonSolverExecute##DIM##D(PoissonSolver *solver,         \
                                      double *phi,                   \
                                      const double *rhs,             \
                                      const int *ngs_phi,            \
                                      const int *ngs_rhs) {          \
        auto *handle =                                               \
            static_cast<ps::ISolver<double, DIM> *>(solver->handle); \
        auto ngs_phi_gs = ToGridSizes<DIM>(ngs_phi);                 \
        auto ngs_rhs_gs = ToGridSizes<DIM>(ngs_rhs);                 \
        handle->Execute(std::span{phi, solver->grid_size},           \
                        std::span{rhs, solver->grid_size},           \
                        ngs_phi_gs,                                  \
                        ngs_rhs_gs);                                 \
    }                                                                \
                                                                     \
    void PoissonSolverExecuteGPU##DIM##D(PoissonSolver *solver,      \
                                         double *d_phi,              \
                                         const double *d_rhs,        \
                                         const int *ngs_phi,         \
                                         const int *ngs_rhs) {       \
        auto *handle =                                               \
            static_cast<ps::ISolver<double, DIM> *>(solver->handle); \
        auto ngs_phi_gs = ToGridSizes<DIM>(ngs_phi);                 \
        auto ngs_rhs_gs = ToGridSizes<DIM>(ngs_rhs);                 \
        handle->ExecuteGPU(std::span{d_phi, solver->grid_size},      \
                           std::span{d_rhs, solver->grid_size},      \
                           ngs_phi_gs,                               \
                           ngs_rhs_gs);                              \
    }

#define DEFINE_FINALIZE(DIM)                                            \
    void PoissonSolverFinalize##DIM##D(PoissonSolver *solver) {         \
        delete static_cast<ps::ISolver<double, DIM> *>(solver->handle); \
    }

#define DEFINE_EXECUTE_FLOAT(DIM)                                              \
    void PoissonSolverExecute##DIM##DFloat(PoissonSolver *solver,              \
                                           float *phi,                         \
                                           const float *rhs,                   \
                                           const int *ngs_phi,                 \
                                           const int *ngs_rhs) {               \
        auto *handle = static_cast<ps::ISolver<float, DIM> *>(solver->handle); \
        auto ngs_phi_gs = ToGridSizes<DIM>(ngs_phi);                           \
        auto ngs_rhs_gs = ToGridSizes<DIM>(ngs_rhs);                           \
        handle->Execute(std::span{phi, solver->grid_size},                     \
                        std::span{rhs, solver->grid_size},                     \
                        ngs_phi_gs,                                            \
                        ngs_rhs_gs);                                           \
    }                                                                          \
                                                                               \
    void PoissonSolverExecuteGPU##DIM##DFloat(PoissonSolver *solver,           \
                                              float *d_phi,                    \
                                              const float *d_rhs,              \
                                              const int *ngs_phi,              \
                                              const int *ngs_rhs) {            \
        auto *handle = static_cast<ps::ISolver<float, DIM> *>(solver->handle); \
        auto ngs_phi_gs = ToGridSizes<DIM>(ngs_phi);                           \
        auto ngs_rhs_gs = ToGridSizes<DIM>(ngs_rhs);                           \
        handle->ExecuteGPU(std::span{d_phi, solver->grid_size},                \
                           std::span{d_rhs, solver->grid_size},                \
                           ngs_phi_gs,                                         \
                           ngs_rhs_gs);                                        \
    }

#define DEFINE_FINALIZE_FLOAT(DIM)                                     \
    void PoissonSolverFinalize##DIM##DFloat(PoissonSolver *solver) {   \
        delete static_cast<ps::ISolver<float, DIM> *>(solver->handle); \
    }

namespace {
DEFINE_EXECUTE(1)
DEFINE_EXECUTE(2)
DEFINE_EXECUTE(3)

DEFINE_FINALIZE(1)
DEFINE_FINALIZE(2)
DEFINE_FINALIZE(3)

DEFINE_EXECUTE_FLOAT(1)
DEFINE_EXECUTE_FLOAT(2)
DEFINE_EXECUTE_FLOAT(3)

DEFINE_FINALIZE_FLOAT(1)
DEFINE_FINALIZE_FLOAT(2)
DEFINE_FINALIZE_FLOAT(3)
}  // namespace

#undef DEFINE_EXECUTE
#undef DEFINE_FINALIZE
#undef DEFINE_EXECUTE_FLOAT
#undef DEFINE_FINALIZE_FLOAT

extern "C" {

// === Uniform Versions ===
// == Double Version ==

void PoissonSolver1DNew(PoissonSolver *solver_out,
                        const int ns[1],
                        const double lens[1],
                        const BoundaryCondition bcs[2],
                        const ps::IApproximation<double> &approx,
                        const int gns[1],
                        const int offs[1],
                        const void *mpi_comm) {
    ps::BCPair bcp = ToCxxBCPair(bcs[0], bcs[1]);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<double, 1> *solver_handle =
        new ps::Solver<double, 1>({static_cast<std::size_t>(ns[0])},
                                  {lens[0]},
                                  {bcp},
                                  {static_cast<std::size_t>(gns[0])},
                                  {static_cast<std::size_t>(offs[0])},
                                  *static_cast<const MPI_Comm *>(mpi_comm),
                                  approx);
#else
    ps::ISolver<double, 1> *solver_handle = new ps::Solver<double, 1>(
        {static_cast<std::size_t>(ns[0])}, {lens[0]}, {bcp}, approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 1,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = 1,
                   .nz = 1};
}

void PoissonSolver2DNew(PoissonSolver *solver_out,
                        const int ns[2],
                        const double lens[2],
                        const BoundaryCondition bcs[4],
                        const ps::IApproximation<double> &approx,
                        const int gns[2],
                        const int offs[2],
                        const void *mpi_comm) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<double, 2> *solver_handle = new ps::Solver<double, 2>(
        {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
        {lens[0], lens[1]},
        {bcp1, bcp2},
        {static_cast<std::size_t>(gns[0]), static_cast<std::size_t>(gns[1])},
        {static_cast<std::size_t>(offs[0]), static_cast<std::size_t>(offs[1])},
        *static_cast<const MPI_Comm *>(mpi_comm),
        approx);
#else
    ps::ISolver<double, 2> *solver_handle = new ps::Solver<double, 2>(
        {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
        {lens[0], lens[1]},
        {bcp1, bcp2},
        approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 2,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = 1};
}

void PoissonSolver3DNew(PoissonSolver *solver_out,
                        const int ns[3],
                        const double lens[3],
                        const BoundaryCondition bcs[6],
                        const ps::IApproximation<double> &approx,
                        const int gns[3],
                        const int offs[3],
                        const void *mpi_comm) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::BCPair bcp3 = ToCxxBCPair(bcs[4], bcs[5]);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<double, 3> *solver_handle =
        new ps::Solver<double, 3>({static_cast<std::size_t>(ns[0]),
                                   static_cast<std::size_t>(ns[1]),
                                   static_cast<std::size_t>(ns[2])},
                                  {lens[0], lens[1], lens[2]},
                                  {bcp1, bcp2, bcp3},
                                  {static_cast<std::size_t>(gns[0]),
                                   static_cast<std::size_t>(gns[1]),
                                   static_cast<std::size_t>(gns[2])},
                                  {static_cast<std::size_t>(offs[0]),
                                   static_cast<std::size_t>(offs[1]),
                                   static_cast<std::size_t>(offs[2])},
                                  *static_cast<const MPI_Comm *>(mpi_comm),
                                  approx);
#else
    ps::ISolver<double, 3> *solver_handle =
        new ps::Solver<double, 3>({static_cast<std::size_t>(ns[0]),
                                   static_cast<std::size_t>(ns[1]),
                                   static_cast<std::size_t>(ns[2])},
                                  {lens[0], lens[1], lens[2]},
                                  {bcp1, bcp2, bcp3},
                                  approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 3,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = static_cast<std::size_t>(ns[2])};
}

void PoissonSolverNew(PoissonSolver *solver_out,
                      int dims,
                      const int *ns,
                      const double *lens,
                      const BoundaryCondition *bcs,
                      Approximation approx,
                      const int *gns,
                      const int *offs,
                      const void *mpi_comm) {
    std::unique_ptr<ps::IApproximation<double>> ps_approx =
        GetApprox<double>(approx);

    switch (dims) {
        case 1:
            PoissonSolver1DNew(
                solver_out, ns, lens, bcs, *ps_approx, gns, offs, mpi_comm);
            break;
        case 2:
            PoissonSolver2DNew(
                solver_out, ns, lens, bcs, *ps_approx, gns, offs, mpi_comm);
            break;
        case 3:
            PoissonSolver3DNew(
                solver_out, ns, lens, bcs, *ps_approx, gns, offs, mpi_comm);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(dims) + ".");
    }
}

void PoissonSolverExecute(PoissonSolver *solver,
                          double *phi,
                          const double *rhs,
                          const int *ngs_phi,
                          const int *ngs_rhs) {
    switch (solver->dim) {
        case 1:
            PoissonSolverExecute1D(solver, phi, rhs, ngs_phi, ngs_rhs);
            break;
        case 2:
            PoissonSolverExecute2D(solver, phi, rhs, ngs_phi, ngs_rhs);
            break;
        case 3:
            PoissonSolverExecute3D(solver, phi, rhs, ngs_phi, ngs_rhs);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(solver->dim) + ".");
    }
}

void PoissonSolverExecuteGPU(PoissonSolver *solver,
                             double *d_phi,
                             const double *d_rhs,
                             const int *ngs_phi,
                             const int *ngs_rhs) {
    switch (solver->dim) {
        case 1:
            PoissonSolverExecuteGPU1D(solver, d_phi, d_rhs, ngs_phi, ngs_rhs);
            break;
        case 2:
            PoissonSolverExecuteGPU2D(solver, d_phi, d_rhs, ngs_phi, ngs_rhs);
            break;
        case 3:
            PoissonSolverExecuteGPU3D(solver, d_phi, d_rhs, ngs_phi, ngs_rhs);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(solver->dim) + ".");
    }
}

void PoissonSolverFinalize(PoissonSolver *solver) {
    switch (solver->dim) {
        case 1:
            PoissonSolverFinalize1D(solver);
            break;
        case 2:
            PoissonSolverFinalize2D(solver);
            break;
        case 3:
            PoissonSolverFinalize3D(solver);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(solver->dim) + ".");
    }
    solver->handle = nullptr;
    solver->grid_size = 0;
}

// == Float Versions ==

void PoissonSolver1DNewFloat(PoissonSolver *solver_out,
                             const int ns[1],
                             const float lens[1],
                             const BoundaryCondition bcs[2],
                             const ps::IApproximation<float> &approx,
                             const int *gns,
                             const int *offs,
                             const void *mpi_comm) {
    ps::BCPair bcp = ToCxxBCPair(bcs[0], bcs[1]);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<float, 1> *solver_handle =
        new ps::Solver<float, 1>({static_cast<std::size_t>(ns[0])},
                                 {lens[0]},
                                 {bcp},
                                 {static_cast<std::size_t>(gns[0])},
                                 {static_cast<std::size_t>(offs[0])},
                                 *static_cast<const MPI_Comm *>(mpi_comm),
                                 approx);
#else
    ps::ISolver<float, 1> *solver_handle = new ps::Solver<float, 1>(
        {static_cast<std::size_t>(ns[0])}, {lens[0]}, {bcp}, approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 1,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = 1,
                   .nz = 1};
}

void PoissonSolver2DNewFloat(PoissonSolver *solver_out,
                             const int ns[2],
                             const float lens[2],
                             const BoundaryCondition bcs[4],
                             const ps::IApproximation<float> &approx,
                             const int *gns,
                             const int *offs,
                             const void *mpi_comm) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<float, 2> *solver_handle = new ps::Solver<float, 2>(
        {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
        {lens[0], lens[1]},
        {bcp1, bcp2},
        {static_cast<std::size_t>(gns[0]), static_cast<std::size_t>(gns[1])},
        {static_cast<std::size_t>(offs[0]), static_cast<std::size_t>(offs[1])},
        *static_cast<const MPI_Comm *>(mpi_comm),
        approx);
#else
    ps::ISolver<float, 2> *solver_handle = new ps::Solver<float, 2>(
        {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
        {lens[0], lens[1]},
        {bcp1, bcp2},
        approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 2,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = 1};
}

void PoissonSolver3DNewFloat(PoissonSolver *solver_out,
                             const int ns[3],
                             const float lens[3],
                             const BoundaryCondition bcs[6],
                             const ps::IApproximation<float> &approx,
                             const int *gns,
                             const int *offs,
                             const void *mpi_comm) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::BCPair bcp3 = ToCxxBCPair(bcs[4], bcs[5]);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<float, 3> *solver_handle =
        new ps::Solver<float, 3>({static_cast<std::size_t>(ns[0]),
                                  static_cast<std::size_t>(ns[1]),
                                  static_cast<std::size_t>(ns[2])},
                                 {lens[0], lens[1], lens[2]},
                                 {bcp1, bcp2, bcp3},
                                 {static_cast<std::size_t>(gns[0]),
                                  static_cast<std::size_t>(gns[1]),
                                  static_cast<std::size_t>(gns[2])},
                                 {static_cast<std::size_t>(offs[0]),
                                  static_cast<std::size_t>(offs[1]),
                                  static_cast<std::size_t>(offs[2])},
                                 *static_cast<const MPI_Comm *>(mpi_comm),
                                 approx);
#else
    ps::ISolver<float, 3> *solver_handle =
        new ps::Solver<float, 3>({static_cast<std::size_t>(ns[0]),
                                  static_cast<std::size_t>(ns[1]),
                                  static_cast<std::size_t>(ns[2])},
                                 {lens[0], lens[1], lens[2]},
                                 {bcp1, bcp2, bcp3},
                                 approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 3,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = static_cast<std::size_t>(ns[2])};
}

void PoissonSolverNewFloat(PoissonSolver *solver_out,
                           int dims,
                           const int *ns,
                           const float *lens,
                           const BoundaryCondition *bcs,
                           Approximation approx,
                           const int *gns,
                           const int *offs,
                           const void *mpi_comm) {
    std::unique_ptr<ps::IApproximation<float>> ps_approx =
        GetApprox<float>(approx);

    switch (dims) {
        case 1:
            PoissonSolver1DNewFloat(
                solver_out, ns, lens, bcs, *ps_approx, gns, offs, mpi_comm);
            break;
        case 2:
            PoissonSolver2DNewFloat(
                solver_out, ns, lens, bcs, *ps_approx, gns, offs, mpi_comm);
            break;
        case 3:
            PoissonSolver3DNewFloat(
                solver_out, ns, lens, bcs, *ps_approx, gns, offs, mpi_comm);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(dims) + ".");
    }
}

void PoissonSolverExecuteFloat(PoissonSolver *solver,
                               float *phi,
                               const float *rhs,
                               const int *ngs_phi,
                               const int *ngs_rhs) {
    switch (solver->dim) {
        case 1:
            PoissonSolverExecute1DFloat(solver, phi, rhs, ngs_phi, ngs_rhs);
            break;
        case 2:
            PoissonSolverExecute2DFloat(solver, phi, rhs, ngs_phi, ngs_rhs);
            break;
        case 3:
            PoissonSolverExecute3DFloat(solver, phi, rhs, ngs_phi, ngs_rhs);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(solver->dim) + ".");
    }
}

void PoissonSolverExecuteGPUFloat(PoissonSolver *solver,
                                  float *d_phi,
                                  const float *d_rhs,
                                  const int *ngs_phi,
                                  const int *ngs_rhs) {
    switch (solver->dim) {
        case 1:
            PoissonSolverExecuteGPU1DFloat(solver, d_phi, d_rhs, ngs_phi, ngs_rhs);
            break;
        case 2:
            PoissonSolverExecuteGPU2DFloat(solver, d_phi, d_rhs, ngs_phi, ngs_rhs);
            break;
        case 3:
            PoissonSolverExecuteGPU3DFloat(solver, d_phi, d_rhs, ngs_phi, ngs_rhs);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(solver->dim) + ".");
    }
}

void PoissonSolverFinalizeFloat(PoissonSolver *solver) {
    switch (solver->dim) {
        case 1:
            PoissonSolverFinalize1DFloat(solver);
            break;
        case 2:
            PoissonSolverFinalize2DFloat(solver);
            break;
        case 3:
            PoissonSolverFinalize3DFloat(solver);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions" +
                                        std::to_string(solver->dim) + ".");
    }
    solver->handle = nullptr;
    solver->grid_size = 0;
}

// === Nonuniform Versions ===
// == Double Versions ==
void PoissonSolverNonuniformZNew(PoissonSolver *solver_out,
                                 const int *ns,
                                 const double *lens,
                                 const double *z,
                                 const double *z_u,
                                 const BoundaryCondition *bcs,
                                 Approximation approx,
                                 const int *gns,
                                 const int *offs,
                                 const void *mpi_comm) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::BCPair bcp3 = ToCxxBCPair(bcs[4], bcs[5]);

    ps::GridSizes<3> grid_sizes = {static_cast<std::size_t>(ns[0]),
                                   static_cast<std::size_t>(ns[1]),
                                   static_cast<std::size_t>(ns[2])};

    auto ps_approx = GetApprox<double>(approx);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<double, 3> *solver_handle = new ps::NonuniformZSolver<double>(
        grid_sizes,
        {lens[0], lens[1]},
        std::span{z, static_cast<std::size_t>(gns[2])},
        std::span{z_u, static_cast<std::size_t>(gns[2]) + 1},
        {bcp1, bcp2, bcp3},
        {static_cast<std::size_t>(gns[0]),
         static_cast<std::size_t>(gns[1]),
         static_cast<std::size_t>(gns[2])},
        {static_cast<std::size_t>(offs[0]),
         static_cast<std::size_t>(offs[1]),
         static_cast<std::size_t>(offs[2])},
        *static_cast<const MPI_Comm *>(mpi_comm),
        *ps_approx);
#else
    ps::ISolver<double, 3> *solver_handle =
        new ps::NonuniformZSolver<double>(grid_sizes,
                                          {lens[0], lens[1]},
                                          std::span{z, grid_sizes[2]},
                                          std::span{z_u, grid_sizes[2] + 1},
                                          {bcp1, bcp2, bcp3},
                                          *ps_approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 3,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = static_cast<std::size_t>(ns[2])};
}

void PoissonSolverNonuniformZExecute(PoissonSolver *solver,
                                     double *phi,
                                     const double *rhs,
                                     const int *ngs_phi,
                                     const int *ngs_rhs) {
    auto *handle = static_cast<ps::ISolver<double, 3> *>(solver->handle);

    auto ngs_phi_gs = ToGridSizes<3>(ngs_phi);
    auto ngs_rhs_gs = ToGridSizes<3>(ngs_rhs);

    handle->Execute(std::span{phi, solver->grid_size},
                    std::span{rhs, solver->grid_size},
                    ngs_phi_gs,
                    ngs_rhs_gs);
}

void PoissonSolverNonuniformZExecuteGPU(PoissonSolver *solver,
                                        double *d_phi,
                                        const double *d_rhs,
                                        const int *ngs_phi,
                                        const int *ngs_rhs) {
    auto *handle = static_cast<ps::ISolver<double, 3> *>(solver->handle);

    auto ngs_phi_gs = ToGridSizes<3>(ngs_phi);
    auto ngs_rhs_gs = ToGridSizes<3>(ngs_rhs);

    handle->ExecuteGPU(std::span{d_phi, solver->grid_size},
                       std::span{d_rhs, solver->grid_size},
                       ngs_phi_gs,
                       ngs_rhs_gs);
}

void PoissonSolverNonuniformZFinalize(PoissonSolver *solver) {
    delete static_cast<ps::ISolver<double, 3> *>(solver->handle);
    solver->handle = nullptr;
    solver->grid_size = 0;
}

// == Float Versions ==
void PoissonSolverNonuniformZNewFloat(PoissonSolver *solver_out,
                                      const int *ns,
                                      const float *lens,
                                      const float *z,
                                      const float *z_u,
                                      const BoundaryCondition *bcs,
                                      Approximation approx,
                                      const int *gns,
                                      const int *offs,
                                      const void *mpi_comm) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::BCPair bcp3 = ToCxxBCPair(bcs[4], bcs[5]);

    ps::GridSizes<3> grid_sizes = {static_cast<std::size_t>(ns[0]),
                                   static_cast<std::size_t>(ns[1]),
                                   static_cast<std::size_t>(ns[2])};

    auto ps_approx = GetApprox<float>(approx);

#ifdef POISSON_SOLVER_USE_MPI
    ps::ISolver<float, 3> *solver_handle = new ps::NonuniformZSolver<float>(
        grid_sizes,
        {lens[0], lens[1]},
        std::span{z, static_cast<std::size_t>(gns[2])},
        std::span{z_u, static_cast<std::size_t>(gns[2]) + 1},
        {bcp1, bcp2, bcp3},
        {static_cast<std::size_t>(gns[0]),
         static_cast<std::size_t>(gns[1]),
         static_cast<std::size_t>(gns[2])},
        {static_cast<std::size_t>(offs[0]),
         static_cast<std::size_t>(offs[1]),
         static_cast<std::size_t>(offs[2])},
        *static_cast<const MPI_Comm *>(mpi_comm),
        *ps_approx);
#else
    ps::ISolver<float, 3> *solver_handle =
        new ps::NonuniformZSolver<float>(grid_sizes,
                                         {lens[0], lens[1]},
                                         std::span{z, grid_sizes[2]},
                                         std::span{z_u, grid_sizes[2] + 1},
                                         {bcp1, bcp2, bcp3},
                                         *ps_approx);
#endif  // POISSON_SOLVER_USE_MPI

    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .dim = 3,
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = static_cast<std::size_t>(ns[2])};
}

void PoissonSolverNonuniformZExecuteFloat(PoissonSolver *solver,
                                          float *phi,
                                          const float *rhs,
                                          const int *ngs_phi,
                                          const int *ngs_rhs) {
    auto *handle = static_cast<ps::ISolver<float, 3> *>(solver->handle);

    auto ngs_phi_gs = ToGridSizes<3>(ngs_phi);
    auto ngs_rhs_gs = ToGridSizes<3>(ngs_rhs);

    handle->Execute(std::span{phi, solver->grid_size},
                    std::span{rhs, solver->grid_size},
                    ngs_phi_gs,
                    ngs_rhs_gs);
}

void PoissonSolverNonuniformZExecuteGPUFloat(PoissonSolver *solver,
                                             float *d_phi,
                                             const float *d_rhs,
                                             const int *ngs_phi,
                                             const int *ngs_rhs) {
    auto *handle = static_cast<ps::ISolver<float, 3> *>(solver->handle);

    auto ngs_phi_gs = ToGridSizes<3>(ngs_phi);
    auto ngs_rhs_gs = ToGridSizes<3>(ngs_rhs);

    handle->ExecuteGPU(std::span{d_phi, solver->grid_size},
                       std::span{d_rhs, solver->grid_size},
                       ngs_phi_gs,
                       ngs_rhs_gs);
}

void PoissonSolverNonuniformZFinalizeFloat(PoissonSolver *solver) {
    delete static_cast<ps::ISolver<float, 3> *>(solver->handle);
    solver->handle = nullptr;
    solver->grid_size = 0;
}
}

#ifdef POISSON_SOLVER_USE_MPI
void PoissonSolverInitMPIGrid(MPI_Comm comm_old,
                              const int *nps,
                              MPI_Comm *comm_out) {
    using poisson_solver::mpi_utils::InitMPIGrid;

    std::array<int, 2> nps_arr{nps[0], nps[1]};
    *comm_out = InitMPIGrid(comm_old, nps_arr);
}

void PoissonSolverGetLocalGridInfo(const int *gns,
                                   MPI_Comm cart_comm,
                                   int *ns,
                                   int *offs) {
    using poisson_solver::mpi_utils::GetLocalGridInfo;

    poisson_solver::GridSizes<3> gns_arr{static_cast<std::size_t>(gns[0]),
                                         static_cast<std::size_t>(gns[1]),
                                         static_cast<std::size_t>(gns[2])};
    auto info = GetLocalGridInfo(gns_arr, cart_comm);
    for (int i = 0; i < 3; ++i) {
        ns[i] = static_cast<int>(info.ns[i]);
        offs[i] = static_cast<int>(info.offs[i]);
    }
}

void PoissonSolverSelectLocalDevice(MPI_Comm comm, int *device_out) {
    using poisson_solver::mpi_utils::SelectLocalDevice;

    *device_out = SelectLocalDevice(comm);
}
#endif  // POISSON_SOLVER_USE_MPI
