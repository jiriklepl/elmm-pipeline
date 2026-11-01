#include <cuda_runtime.h>
#include <cstddef>

extern "C" {
    void* fortran_cuda_malloc(std::size_t bytes) {
        void* ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }

    void fortran_cuda_free(void* ptr) {
        cudaFree(ptr);
    }

    void fortran_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }

    void fortran_cuda_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
}
