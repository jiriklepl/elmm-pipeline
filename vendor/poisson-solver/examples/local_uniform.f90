program local_uniform
    use iso_c_binding
    use PoisSolver

    implicit none

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

    ! The buffers hold the grid and the ghost cells on both sides.
    ! There is no ngs argument on the Fortran side: poisson_solver_execute
    ! reads the ghost-cell count back out of the array bounds, so the shape
    ! alone must carry it.
    real(c_float) :: rhs(1 - ngs : side_size + ngs, &
                         1 - ngs : side_size + ngs, &
                         1 - ngs : side_size + ngs)
    real(c_float) :: phi(1 - ngs : side_size + ngs, &
                         1 - ngs : side_size + ngs, &
                         1 - ngs : side_size + ngs)

    type(PoissonSolver) :: solver

    ! Initialize the solver
    call poisson_solver_new_float(solver, ndims, ns, lens, bcs, &
        PoisSolver_FiniteDifference2)

    ! Prepare input
    ! Fill it with desired values (here, we fill 1.0)
    rhs = 1.0_c_float

    ! Prepare result buffer
    phi = 0.0_c_float

    ! Solve the equation
    ! (phi and rhs have the same amount of ghost cells)
    call poisson_solver_execute_float(solver, phi, rhs)

    ! Use double precision instead by dropping the _float suffix on every
    ! call (poisson_solver_new, poisson_solver_execute,
    ! poisson_solver_finalize)

    ! Explicit finalization
    call poisson_solver_finalize_float(solver)

    write (*, *) "Successfully finished example local_uniform"

end program local_uniform
