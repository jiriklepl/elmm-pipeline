#ifndef C_BINDING_H
#define C_BINDING_H

// Turn off C++ modernization rules for this pure C header
// NOLINTBEGIN(modernize-*)

#include <stddef.h>
#include <stdint.h>

#ifdef POISSON_SOLVER_USE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct PoissonSolver {
    void *handle;
    size_t grid_size;
    size_t dim;
    size_t nx, ny, nz;
} PoissonSolver;

typedef enum {
    Periodic,
    Dirichlet,
    Neumann,
    DirichletStag,
    NeumannStag
} BoundaryCondition;

typedef enum { Spectral = 0, FD2 = 2, FD4 = 4 } Approximation;

// === Uniform Solvers ===
/**
 * @brief Constructs a poisson solver with double precision.
 *
 * @param[out] solver_out
 * @param[in] dims Dimensionality of the solver.
 * @param[in] ns Local sizes of the grid in each dimension (without ghost cells).
 * @param[in] lens Real lengths of the grid in each dimension.
 * @param[in] bcs Boundary conditions for each axis in the order {{west, east},
 * {south, north}, {bottom, top}}.
 * @param[in] approx Desired approximation.
 * @param[in] gns Global sizes of the grid in each dimension (without ghost
 * cells).
 * @param[in] offs Offset of local data.
 * @param mpi_comm The MPI communicator.
 *
 * @note The solver assumes that the first index is the fastest changing
 * and the last is the slowest.
 *
 * @note \p gns, \p offs and \p mpi_comm are not used if POISSON_SOLVER_USE_MPI is disabled.
 */
void PoissonSolverNew(PoissonSolver *solver_out,
                      int dims,
                      const int *ns,
                      const double *lens,
                      const BoundaryCondition *bcs,
                      Approximation approx,
                      const int *gns,
                      const int *offs,
                      const void *mpi_comm);

/**
 * @brief Solves the equation \f$ \Delta \phi = b \f$ in double precision.
 *
 * @param[in] solver Solver constructed via PoissonSolverNew().
 * @param[out] phi
 * @param[in] rhs
 * @param[in] ngs_phi Number of ghost cells (on one side)
 * \p phi has in each dimension.
 * @param[in] ngs_rhs Number of ghost cells (on one side)
 * \p rhs has in each dimension.
 */
void PoissonSolverExecute(PoissonSolver *solver,
                          double *phi,
                          const double *rhs,
                          const int *ngs_phi,
                          const int *ngs_rhs);

/**
 * @brief Same as PoissonSolverExecute() but expects device pointers
 * in \p d_phi and \p d_rhs.
 */
void PoissonSolverExecuteGPU(PoissonSolver *solver,
                             double *d_phi,
                             const double *d_rhs,
                             const int *ngs_phi,
                             const int *ngs_rhs);

/**
 * @brief Frees the solver.
 *
 * @param solver Solver constructed via PoissonSolverNew().
 */
void PoissonSolverFinalize(PoissonSolver *solver);

/**
 * @brief Same as PoissonSolverNew() but uses float precision.
 */
void PoissonSolverNewFloat(PoissonSolver *solver_out,
                           int dims,
                           const int *ns,
                           const float *lens,
                           const BoundaryCondition *bcs,
                           Approximation approx,
                           const int *gns,
                           const int *offs,
                           const void *mpi_comm);

/**
 * @brief Same as PoissonSolverExecute() but used float precision and
 * \p solver must be constructed via PoissonSolverNewFloat().
 */
void PoissonSolverExecuteFloat(PoissonSolver *solver,
                               float *phi,
                               const float *rhs,
                               const int *ngs_phi,
                               const int *ngs_rhs);

/**
 * @brief Same as PoissonSolverExecuteGPU() but uses float precision and
 * \p solver must be constructed via PoissonSolverNewFloat().
 */
void PoissonSolverExecuteGPUFloat(PoissonSolver *solver,
                                  float *d_phi,
                                  const float *d_rhs,
                                  const int *ngs_phi,
                                  const int *ngs_rhs);

/**
 * @brief Same as PoissonSolverFinalize() but \p solver must be constructed
 * via PoissonSolverNewFloat().
 */
void PoissonSolverFinalizeFloat(PoissonSolver *solver);

// === Nonuniform Solvers ===
/**
 * @brief Constructs a poisson solver with nonuniform Z axis
 * and double precision.
 *
 * @param[out] solver_out
 * @param ns Sizes of the grid in each dimension (without ghost cells).
 * @param lens Real lengths of the grid in each dimension.
 * @param z Coordinates representing the cell centers along the Z-axis.
 * Must contain at least `gns[2]` elements
 * (`ns[2]` if POISSON_SOLVER_USE_MPI is disabled).
 * @param z_u Coordinates representing the cell faces along the Z-axis.
 * Must contain at least `gns[2] + 1` elements
 * (`ns[2] + 1` if POISSON_SOLVER_USE_MPI is disabled).
 * @param bcs Boundary conditions for each axis in the order {{west, east},
 * {south, north}, {bottom, top}}.
 * For more information @see boundary_conditions.hpp.
 * @param gns Global sizes of the grid in each dimension (without ghost
 * cells).
 * @param offs Offset of local data.
 * @param comm The MPI communicator.
 * @param approx Desired approximation function. @see approximations.hpp
 *
 * @note The solver assumes that the first index is the fastest changing
 * and the last is the slowest.
 *
 * @note \p gns, \p offs and \p mpi_comm are not used if POISSON_SOLVER_USE_MPI is disabled.
 */
void PoissonSolverNonuniformZNew(PoissonSolver *solver_out,
                                 const int ns[3],
                                 const double lens[2],
                                 const double *z,
                                 const double *z_u,
                                 const BoundaryCondition bcs[6],
                                 Approximation approx,
                                 const int gns[3],
                                 const int offs[3],
                                 const void *mpi_comm);

/**
 * @brief Same as PoissonSolverExecute() but the \p solver must be
 * constructed via PoissonSolverNonuniformZNew().
 */
void PoissonSolverNonuniformZExecute(PoissonSolver *solver,
                                     double *phi,
                                     const double *rhs,
                                     const int ngs_phi[3],
                                     const int ngs_rhs[3]);

/**
 * @brief Same as PoissonSolverExecuteGPU() but the \p solver must be
 * constructed via PoissonSolverNonuniformZNew().
 */
void PoissonSolverNonuniformZExecuteGPU(PoissonSolver *solver,
                                        double *d_phi,
                                        const double *d_rhs,
                                        const int ngs_phi[3],
                                        const int ngs_rhs[3]);

/**
 * @brief Same as PoissonSolverFinalize() but the \p solver must be
 * constructed via PoissonSolverNonuniformZNew().
 */
void PoissonSolverNonuniformZFinalize(PoissonSolver *solver);

/**
 * @brief Same as PoissonSolverNonuniformZNew() but uses float precision.
 */
void PoissonSolverNonuniformZNewFloat(PoissonSolver *solver_out,
                                      const int ns[3],
                                      const float lens[2],
                                      const float *z,
                                      const float *z_u,
                                      const BoundaryCondition bcs[6],
                                      Approximation approx,
                                      const int gns[3],
                                      const int offs[3],
                                      const void *mpi_comm);

/**
 * @brief Same as PoissonSolverExecuteFloat() but the \p solver must be
 * constructed via PoissonSolverNonuniformZNewFloat().
 */
void PoissonSolverNonuniformZExecuteFloat(PoissonSolver *solver,
                                          float *phi,
                                          const float *rhs,
                                          const int ngs_phi[3],
                                          const int ngs_rhs[3]);

/**
 * @brief Same as PoissonSolverExecuteGPUFloat() but the \p solver must be
 * constructed via PoissonSolverNonuniformZNewFloat().
 */
void PoissonSolverNonuniformZExecuteGPUFloat(PoissonSolver *solver,
                                             float *d_phi,
                                             const float *d_rhs,
                                             const int ngs_phi[3],
                                             const int ngs_rhs[3]);

/**
 * @brief Same as PoissonSolverFinalizeFloat() but the \p solver must be
 * constructed via PoissonSolverNonuniformZNewFloat().
 */
void PoissonSolverNonuniformZFinalizeFloat(PoissonSolver *solver);

#ifdef POISSON_SOLVER_USE_MPI
/**
 * @brief Initializes MPI cartesian grid
 * with at most two axes of decomposition (the X axis is local).
 *
 * @param[in] comm_old The old communicator.
 * @param[in] nps Desired number of processes per dimension.
 * nps[0] corresponds to the Y axis and nps[1] to the Z axis.
 * For 1D decomposition, leave the non-decomposed axis as 1.
 * @param[out] comm_out The resulting communicator.
 */
void PoissonSolverInitMPIGrid(MPI_Comm comm_old,
                              const int nps[2],
                              MPI_Comm *comm_out);

/**
 * @brief Retrieves information about the local grid.
 *
 * @param[in] gns Global sizes of the grid in each dimension.
 * @param[in] cart_comm Communicator created via a call to InitMPIGrid.
 * @param[out] ns Local grid sizes.
 * @param[out] offs Offset of local grid from the origin
 */
void PoissonSolverGetLocalGridInfo(const int gns[3],
                                   MPI_Comm cart_comm,
                                   int ns[3],
                                   int offs[3]);

/**
 * @brief Selects one CUDA device per rank based on this process's rank
 * within its node, and makes it the active device.
 *
 * @param[in] comm A communicator containing (at least) all ranks that
 * share this node.
 * @param[out] device_out The CUDA device ordinal selected for this rank.
 */
void PoissonSolverSelectLocalDevice(MPI_Comm comm, int *device_out);
#endif

#ifdef __cplusplus
}
#endif

// NOLINTEND(modernize-*)

#endif  // C_BINDING_H
