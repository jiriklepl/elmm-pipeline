program local_uniform_nocopy
    use iso_c_binding
    use PoisSolver

    implicit none

    ! A small interoperability layer around cudaMalloc, cudaFree and
    ! cudaMemcpy. The library itself exposes no device-memory API to
    ! Fortran, so a caller that wants device buffers must provide one.
    ! We borrow the one used by poisson-solver Fortran tests.
    ! You can find it in tests/include/cuda_fortran_helpers.cpp
    interface
        function fortran_cuda_malloc(bytes) bind(C, name="fortran_cuda_malloc")
            import :: c_size_t, c_ptr
            integer(c_size_t), value :: bytes
            type(c_ptr) :: fortran_cuda_malloc
        end function fortran_cuda_malloc

        subroutine fortran_cuda_free(ptr) bind(C, name="fortran_cuda_free")
            import :: c_ptr
            type(c_ptr), value :: ptr
        end subroutine fortran_cuda_free

        subroutine fortran_cuda_memcpy_h2d(dst, src, bytes) &
                bind(C, name="fortran_cuda_memcpy_h2d")
            import :: c_size_t, c_ptr
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: bytes
        end subroutine fortran_cuda_memcpy_h2d

        subroutine fortran_cuda_memcpy_d2h(dst, src, bytes) &
                bind(C, name="fortran_cuda_memcpy_d2h")
            import :: c_size_t, c_ptr
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: bytes
        end subroutine fortran_cuda_memcpy_d2h
    end interface

    ! We will construct a 3D solver
    integer, parameter :: ndims = 3

    ! Define the grid side sizes
    integer, parameter :: side_size = 64
    integer :: ns(ndims) = [side_size, side_size, side_size]

    ! Define the grid side lengths
    real(c_float) :: lens(ndims) = [1.0_c_float, 1.0_c_float, 1.0_c_float]

    ! Define the boundary conditions
    ! in the order of {west, east, south, north, bottom, top}
    integer :: bcs(2 * ndims) = [ &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag, &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag, &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag]

    ! Grid will have two ghost cells on every side
    ! One axis will therefore have four ghost cells
    ! (two on the left, two on the right)
    integer, parameter :: ngs = 2

    ! The buffers hold the grid and the ghost cells on both sides
    real(c_float), target :: rhs(1 - ngs : side_size + ngs, &
                                 1 - ngs : side_size + ngs, &
                                 1 - ngs : side_size + ngs)
    real(c_float), target :: phi(1 - ngs : side_size + ngs, &
                                 1 - ngs : side_size + ngs, &
                                 1 - ngs : side_size + ngs)
    integer(c_size_t) :: bytes

    ! Allocate the input and result buffers directly on the device.
    type(c_ptr) :: d_phi_ptr, d_rhs_ptr
    real(c_float), pointer :: d_phi(:, :, :), d_rhs(:, :, :)

    type(PoissonSolver) :: solver

    ! Initialize the solver
    call poisson_solver_new_float(solver, ndims, ns, lens, bcs, &
        PoisSolver_FiniteDifference2)

    bytes = size(rhs, kind=c_size_t) * &
        c_sizeof(rhs(1 - ngs, 1 - ngs, 1 - ngs))

    d_phi_ptr = fortran_cuda_malloc(bytes)
    d_rhs_ptr = fortran_cuda_malloc(bytes)
    call c_f_pointer(d_phi_ptr, d_phi, shape(phi))
    call c_f_pointer(d_rhs_ptr, d_rhs, shape(rhs))

    ! A real application would already have the right-hand side on the
    ! device (for example, produced there by a previous kernel). Here we
    ! fill a host buffer instead and copy it over, purely to have some
    ! input data for the example.
    rhs = 1.0_c_float
    call fortran_cuda_memcpy_h2d( &
        d_rhs_ptr, c_loc(rhs(1 - ngs, 1 - ngs, 1 - ngs)), bytes)

    ! Solve the equation directly on the device buffers.
    ! No host<->device copy of phi or rhs happens inside the solver call
    ! itself.
    ! (phi and rhs have the same amount of ghost cells)
    call poisson_solver_execute_gpu_float(solver, d_phi, d_rhs)

    ! Copy the result back to the host to inspect it.
    phi = 0.0_c_float
    call fortran_cuda_memcpy_d2h( &
        c_loc(phi(1 - ngs, 1 - ngs, 1 - ngs)), d_phi_ptr, bytes)

    ! Explicit finalization
    call poisson_solver_finalize_float(solver)

    call fortran_cuda_free(d_phi_ptr)
    call fortran_cuda_free(d_rhs_ptr)

    write (*, *) "Successfully finished example local_uniform_nocopy"

end program local_uniform_nocopy
