# Fetch doctest
include(FetchContent)
set(DOCTEST_VERSION v2.5.2)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest
    GIT_TAG 6804767ee637789db8a5cb281381cae98dc36906
)
FetchContent_MakeAvailable(doctest)

set(DOCTEST_INCLUDE_DIR ${doctest_SOURCE_DIR})

# Include doctest_discover_tests(<target>) function
include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

# Prepare variables for launching via srun
set(SLURM_LAUNCH_BASE
    srun
    -p gpu-short-teach
    -w "volta[02-05]"
)
set(SLURM_FLAG_GPU
    --gpus-per-task 1
    # Each process has its own gpu
    --gpu-bind=single:1
)
set(SLURM_FLAG_LOCAL
    -N 1
    --ntasks-per-node 1
)
set(SLURM_FLAG_DISTRIBUTED
    --mpi=pmix
    -N 2
    --ntasks-per-node 2
)
set(SLURM_FLAG_GPU_DISTRIBUTED
    --gpus 4
)
set(CH_CLD_LAUNCH
    ch-run
    # The mapped directory
    -b ../../../../:/opt/build
    ../../../../../imgdir --
)
set(LOCAL_TEST_LAUNCH
    ${SLURM_LAUNCH_BASE}
    ${SLURM_FLAG_GPU}
    ${SLURM_FLAG_LOCAL}
    ${CH_CLD_LAUNCH}
)
set(DISTRIBUTED_TEST_LAUNCH
    ${SLURM_LAUNCH_BASE}
    ${SLURM_FLAG_GPU_DISTRIBUTED}
    ${SLURM_FLAG_DISTRIBUTED}
    ${CH_CLD_LAUNCH}
)

# == C/C++ Residuum Tests ==
if (POISSON_SOLVER_USE_MPI)
    # TODO - Add sources for CXX MPI tests
    set(CXX_TEST_SRC "")
else()
    set(CXX_TEST_SRC
        test_main.cpp
        test_residuum.cpp
        test_residuum_nonuniform_z.cpp
    )
endif()

# TODO - Remove "if" statement when the CXX_TEST_SRC is
# guaranteed to be always populated (i.e., when CXX MPI tests are rdy)
if (CXX_TEST_SRC)
    # By default, only the lib is built
    add_executable(residuum-tests EXCLUDE_FROM_ALL ${CXX_TEST_SRC})

    target_include_directories(residuum-tests PRIVATE
        ../include/
    )
    target_link_libraries(residuum-tests PRIVATE
        common_flags
        poisson-solver
        doctest::doctest
    )

    # We cannot use `doctest_discover_tests` as that will
    # allocate a worker node with a GPU every time the tests
    # are built.
    add_test(
        NAME residuum-tests-all
        COMMAND
        ${LOCAL_TEST_LAUNCH}
        $<TARGET_FILE:residuum-tests>
    )

    # TODO - MPI test launching via srun is unhandled here
    # Add when CXX MPI tests are present
endif()

# == Fortran Residuum Tests ==
if (POISSON_SOLVER_USE_MPI)
    set(FORTRAN_TEST_SRCS
        test_residuum_mpi.f90
        test_residuum_mpi_nonuniform_z.f90
    )
else()
    set(FORTRAN_TEST_SRCS
        test_residuum.f90
        test_residuum_nonuniform_z.f90
    )
endif()

set(ALL_FORTRAN_TEST_TARGETS "")

foreach(f_test_src IN LISTS FORTRAN_TEST_SRCS)
    # Extract filename
    get_filename_component(base_name ${f_test_src} NAME_WE)

    set(test_target "fortran-${base_name}")

    # By default, only the lib is built
    add_executable(${test_target} EXCLUDE_FROM_ALL ${f_test_src} ../include/cuda_fortran_helpers.cpp)

    target_include_directories(${test_target} PRIVATE
        ${CMAKE_Fortran_MODULE_DIRECTORY})

    # Automatically links with `stdc++`
    target_link_libraries(${test_target} PRIVATE
        common_flags
        poisson-solver
    )
    if (POISSON_SOLVER_BUILD_FORTRAN AND POISSON_SOLVER_USE_MPI)
        target_link_libraries(poisson-solver PUBLIC
            MPI::MPI_Fortran
        )
    endif()

    # Register test with ctest
    if (POISSON_SOLVER_USE_MPI)
        add_test(
            NAME ${test_target}
            COMMAND
            ${DISTRIBUTED_TEST_LAUNCH}
            $<TARGET_FILE:${test_target}>
        )
        set_tests_properties(${test_target} PROPERTIES
            PROCESSORS 4
            # Make UCX not use ptrace
            # Otherwise many errors get printed before falling back to a different method
            ENVIRONMENT "UCX_POSIX_USE_PROC_LINK=n"
        )
    else()
        add_test(
            NAME ${test_target}
            COMMAND
            ${LOCAL_TEST_LAUNCH}
            $<TARGET_FILE:${test_target}>
        )
    endif()

    # Tests fail when `FAIL` appears on output
    set_tests_properties(${test_target}
        PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL"
    )

    list(APPEND ALL_FORTRAN_TEST_TARGETS ${test_target})
endforeach()

# == Group Tests Together ==
add_custom_target(tests)

# TODO - When CXX MPI tests are rdy, keep only the else branch
if (POISSON_SOLVER_USE_MPI)
    add_dependencies(tests ${ALL_FORTRAN_TEST_TARGETS})
else()
    add_dependencies(tests residuum-tests ${ALL_FORTRAN_TEST_TARGETS})
endif()
