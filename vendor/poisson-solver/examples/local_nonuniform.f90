program local_nonuniform
    use iso_c_binding
    use PoisSolver

    implicit none

    ! We will construct a 3D solver
    integer, parameter :: ndims = 3

    ! Define the grid side sizes
    integer, parameter :: side_size = 64
    integer :: ns(ndims) = [side_size, side_size, side_size]

    ! Define the grid side lengths in X and Y
    ! The Z length instead comes from the z and z_u arrays (see below)
    real(c_float) :: lens(2) = [1.0_c_float, 1.0_c_float]

    ! Define the boundary conditions
    ! in the order of {west, east, south, north, bottom, top}
    integer :: bcs(2 * ndims) = [ &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag, &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag, &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag]

    ! The Z axis is nonuniform, so two arrays describe it.
    ! z_u holds the cell faces: side_size + 1 values, from the bottom up.
    ! z holds the cell centers: side_size values, one per cell.
    ! Cell k reaches from z_u(k) to z_u(k + 1), and z(k) lies between them.
    real(c_float), parameter :: len_z = 1.0_c_float
    real(c_float), parameter :: dz = len_z / side_size
    real(c_float) :: z_u(0:side_size)
    real(c_float) :: z(0:side_size - 1)
    integer :: k

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

    ! Even spacing keeps this example short.
    ! A real case may space the faces unevenly.
    do k = 0, side_size
        z_u(k) = real(k, c_float) * dz
    end do

    do k = 0, side_size - 1
        z(k) = (z_u(k) + z_u(k + 1)) / 2
    end do

    ! Initialize the solver
    call poisson_solver_nonuniform_z_new_float(solver, ns, lens, z, z_u, &
        bcs, PoisSolver_FiniteDifference2)

    ! Prepare input
    ! Fill it with desired values (here, we fill 1.0)
    rhs = 1.0_c_float

    ! Prepare result buffer
    phi = 0.0_c_float

    ! Solve the equation
    ! (phi and rhs have the same amount of ghost cells)
    call poisson_solver_nonuniform_z_execute_float(solver, phi, rhs)

    ! Explicit finalization
    call poisson_solver_nonuniform_z_finalize_float(solver)

    write (*, *) "Successfully finished example local_nonuniform"

end program local_nonuniform
