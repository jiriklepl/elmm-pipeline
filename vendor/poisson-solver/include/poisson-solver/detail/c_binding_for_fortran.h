#include "poisson-solver/c_binding.h"

#ifdef POISSON_SOLVER_USE_MPI
#include <mpi.h>
#endif  // POISSON_SOLVER_USE_MPI

#ifdef __cplusplus
extern "C" {
#endif
void PoissonSolverNewForFortran(PoissonSolver *solver_out,
                                int dims,
                                const int *ns,
                                const double *lens,
                                const BoundaryCondition *bcs,
                                Approximation approx,
                                const int *gns,
                                const int *offs,
#ifdef POISSON_SOLVER_USE_MPI
                                const MPI_Fint *mpi_comm
#else
                                const void *mpi_comm
#endif  // POISSON_SOLVER_USE_MPI
);

void PoissonSolverNewFloatForFortran(PoissonSolver *solver_out,
                                     int dims,
                                     const int *ns,
                                     const float *lens,
                                     const BoundaryCondition *bcs,
                                     Approximation approx,
                                     const int *gns,
                                     const int *offs,
#ifdef POISSON_SOLVER_USE_MPI
                                     const MPI_Fint *mpi_comm
#else
                                     const void *mpi_comm
#endif  // POISSON_SOLVER_USE_MPI
);

void PoissonSolverNonuniformZNewForFortran(PoissonSolver *solver_out,
                                           const int *ns,
                                           const double *lens,
                                           const double *z,
                                           const double *z_u,
                                           const BoundaryCondition *bcs,
                                           Approximation approx,
                                           const int *gns,
                                           const int *offs,
#ifdef POISSON_SOLVER_USE_MPI
                                           const MPI_Fint *mpi_comm
#else
                                           const void *mpi_comm
#endif  // POISSON_SOLVER_USE_MPI
);

void PoissonSolverNonuniformZNewFloatForFortran(PoissonSolver *solver_out,
                                                const int *ns,
                                                const float *lens,
                                                const float *z,
                                                const float *z_u,
                                                const BoundaryCondition *bcs,
                                                Approximation approx,
                                                const int *gns,
                                                const int *offs,
#ifdef POISSON_SOLVER_USE_MPI
                                                const MPI_Fint *mpi_comm
#else
                                                const void *mpi_comm
#endif  // POISSON_SOLVER_USE_MPI
);

#ifdef POISSON_SOLVER_USE_MPI
void PoissonSolverInitMPIGridForFortran(const MPI_Fint *comm_old,
                                        const int *nps,
                                        MPI_Fint *comm_out);

void PoissonSolverGetLocalGridInfoForFortran(const int *gns,
                                             const MPI_Fint *cart_comm,
                                             int *ns,
                                             int *offs);
#endif  // POISSON_SOLVER_USE_MPI

#ifdef __cplusplus
}
#endif
