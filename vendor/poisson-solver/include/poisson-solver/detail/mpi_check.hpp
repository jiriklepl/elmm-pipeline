#ifndef MPI_CHECK_HPP
#define MPI_CHECK_HPP
#ifdef POISSON_SOLVER_USE_MPI

#include <mpi.h>
#include <source_location>
#include <sstream>
#include <stdexcept>

inline void MPICheck(int mpi_err_code, const std::source_location &loc) {
    if (mpi_err_code != MPI_SUCCESS) {
        std::stringstream err_stream{};
        err_stream << "MPI Error (" << mpi_err_code << ")\n"
                   << "at " << loc.file_name() << "[" << loc.line() << "]";
        throw std::runtime_error(err_stream.str());
    }
}

#define MPI_CHECK(err_code) MPICheck(err_code, std::source_location::current())

#endif  // POISSON_SOLVER_USE_MPI
#endif  // MPI_CHECK_HPP
