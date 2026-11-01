if (POISSON_SOLVER_USE_MPI)
    set(BENCH_SRC benchmark_mpi.cpp)
else()
    set(BENCH_SRC benchmark.cpp)
endif()

# By default, only the lib is built
add_executable(benchmarks EXCLUDE_FROM_ALL ${BENCH_SRC})

target_include_directories(benchmarks PRIVATE ../include)
target_link_libraries(benchmarks PRIVATE common_flags poisson-solver)
