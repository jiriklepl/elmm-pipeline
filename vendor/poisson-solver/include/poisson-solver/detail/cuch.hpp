#ifndef CUCH_HPP
#define CUCH_HPP

// Written by Martin Kruliš for CUDA Blur Stencil lab assignment.
// Modified by Matouš Zatloukal.

#include <cuda_runtime.h>
#include <exception>
#include <sstream>
#include <utility>

namespace poisson_solver::detail {
/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception {
public:
    explicit CudaError(cudaError_t status = cudaSuccess)
        : std::exception(),
          status_(status) {}

    explicit CudaError(const char *msg, cudaError_t status = cudaSuccess)
        : std::exception(),
          message_(msg),
          status_(status) {}

    explicit CudaError(std::string msg, cudaError_t status = cudaSuccess)
        : std::exception(),
          message_(std::move(msg)),
          status_(status) {}

    ~CudaError() noexcept override = default;

    auto what() const noexcept -> const char * override {
        return message_.c_str();
    }

    // Overloading << operator that uses stringstream to append data to
    // mMessage.
    template <typename T>
    auto operator<<(const T &data) -> CudaError & {
        std::stringstream stream;
        stream << message_ << data;
        message_ = stream.str();
        return *this;
    }

protected:
    // Internal buffer where the message is kept.
    std::string message_;

    cudaError_t status_;
};

/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void CudaCheck(cudaError_t status,
                      int line,
                      const char *src_file,
                      const char *err_msg = nullptr) {
    if (status != cudaSuccess) {
        throw(CudaError(status)
              << "CUDA Error (" << status << "): " << cudaGetErrorString(status)
              << "\n"
              << "at " << src_file << "[" << line << "]: " << err_msg);
    }
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) \
    ::poisson_solver::detail::CudaCheck(status, __LINE__, __FILE__, #status)
}  // namespace poisson_solver::detail

#endif  // CUCH_HPP
