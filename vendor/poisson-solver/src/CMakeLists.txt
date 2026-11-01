set(LIB_SRC_CXX
    c_binding.cpp
    device_api.cpp
    detail/kernel_binary_cache.cpp
    detail/nvrtc_cache.cpp
    detail/c_binding_for_fortran.cpp
    detail/runners/common.cu
    detail/runners/common_real.cu
    detail/runners/common_cmplx.cu
    detail/runners/nonuniform_z.cu
    mpi_utils.cpp
)
set(FORTRAN_BINDING
    fortran_binding.f90
    device_binding.f90
)


# Debug files shall have a `_debug` suffix
set(CMAKE_DEBUG_POSTFIX "_debug")

# Output STATIC libraries here
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Output SHARED libraries here
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(poisson-solver ${LIB_SRC_CXX})
set_target_properties(poisson-solver PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

if (POISSON_SOLVER_BUILD_FORTRAN)
    target_sources(poisson-solver PRIVATE ${FORTRAN_BINDING})
    target_include_directories(poisson-solver INTERFACE
        $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/poisson-solver/fortran>)
endif()

# The `BUILD_INTERFACE` evals to true when target 
# linking againts this lib is part of the same CMake build tree
# (e.g. called `add_subdirectory(libs/poisson-solver))`.
target_include_directories(poisson-solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(poisson-solver PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(poisson-solver PRIVATE $<BUILD_INTERFACE:common_flags> OpenSSL::Crypto ${CMAKE_DL_LIBS})
target_compile_features(poisson-solver PUBLIC cxx_std_20)
target_link_libraries(poisson-solver PUBLIC CUDA::cudart VkFFT)
if (POISSON_SOLVER_USE_MPI)
    target_link_libraries(poisson-solver PUBLIC
        Heffte
        MPI::MPI_CXX
    )

    # Defines `POISSON_SOLVER_USE_MPI`, therefore code can do `#ifdef POISSON_SOLVER_USE_MPI`
    target_compile_definitions(poisson-solver PUBLIC POISSON_SOLVER_USE_MPI)
endif()

if (POISSON_SOLVER_ENABLE_BENCHMARKS)
    target_compile_definitions(poisson-solver PUBLIC POISSON_SOLVER_ENABLE_BENCHMARKS)
endif()
