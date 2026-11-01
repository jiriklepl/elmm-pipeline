# This CMake uses common_flags target to add compiler options
# This target is defined in the root CMake and is not specific
# to these examples.

# The .c examples need a C compiler. project() at the root only enables
# CXX and CUDA, so the C language is not otherwise available.
enable_language(C)

set(ALL_EXAMPLE_TARGETS "")

# A build is either local or distributed
if (POISSON_SOLVER_USE_MPI)
    set(CXX_C_EXAMPLES
        distributed_uniform
        distributed_nonuniform
    )
    set(FORTRAN_EXAMPLES ${CXX_C_EXAMPLES})
else()
    set(CXX_C_EXAMPLES
        local_uniform
        local_nonuniform
        local_uniform_nocopy
    )
    # local_uniform_nocopy.f90 needs the CUDA helper below, so it is built
    # separately from this list.
    set(FORTRAN_EXAMPLES
        local_uniform
        local_nonuniform
    )
endif()

foreach(name IN LISTS CXX_C_EXAMPLES)
    add_executable(${name}-cpp EXCLUDE_FROM_ALL ${name}.cpp)
    target_link_libraries(${name}-cpp PRIVATE common_flags poisson-solver)
    list(APPEND ALL_EXAMPLE_TARGETS ${name}-cpp)

    add_executable(${name}-c EXCLUDE_FROM_ALL ${name}.c)
    target_link_libraries(${name}-c PRIVATE common_flags poisson-solver)
    list(APPEND ALL_EXAMPLE_TARGETS ${name}-c)
endforeach()

if (NOT POISSON_SOLVER_USE_MPI)
    # local_with_pretty_print exists in C++ only, so it is not part of the
    # per-language lists above.
    add_executable(local_with_pretty_print-cpp EXCLUDE_FROM_ALL
        local_with_pretty_print.cpp)
    target_link_libraries(local_with_pretty_print-cpp PRIVATE
        common_flags poisson-solver)
    list(APPEND ALL_EXAMPLE_TARGETS local_with_pretty_print-cpp)
endif()

if (POISSON_SOLVER_BUILD_FORTRAN)
    foreach(name IN LISTS FORTRAN_EXAMPLES)
        set(target_name ${name}-f90)

        add_executable(${target_name} EXCLUDE_FROM_ALL ${name}.f90)
        target_include_directories(${target_name} PRIVATE
            ${CMAKE_Fortran_MODULE_DIRECTORY})
        target_link_libraries(${target_name} PRIVATE common_flags poisson-solver)

        # Only the distributed Fortran examples need the Fortran MPI module.
        if (POISSON_SOLVER_USE_MPI)
            target_link_libraries(${target_name} PRIVATE MPI::MPI_Fortran)
        endif()

        list(APPEND ALL_EXAMPLE_TARGETS ${target_name})
    endforeach()

    if (NOT POISSON_SOLVER_USE_MPI)
        # local_uniform_nocopy.f90 has no CUDA memory API of its own: it
        # calls the same small CUDA helper the Fortran tests use.
        add_executable(local_uniform_nocopy-f90 EXCLUDE_FROM_ALL
            local_uniform_nocopy.f90
            ../tests/include/cuda_fortran_helpers.cpp
        )
        target_include_directories(local_uniform_nocopy-f90 PRIVATE
            ${CMAKE_Fortran_MODULE_DIRECTORY})
        target_link_libraries(local_uniform_nocopy-f90 PRIVATE
            common_flags poisson-solver)
        list(APPEND ALL_EXAMPLE_TARGETS local_uniform_nocopy-f90)
    endif()
endif()

# == Group Examples Together ==
add_custom_target(examples)
add_dependencies(examples ${ALL_EXAMPLE_TARGETS})
