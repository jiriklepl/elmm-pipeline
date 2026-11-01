program distributed_nonuniform
    use iso_c_binding
    use mpi
    use PoisSolver

    implicit none

    ! We will construct a 3D solver
    integer, parameter :: ndims = 3

    integer :: ie
    integer :: comm_size
    integer :: cart_comm
    integer :: nps(2)
    integer :: local_device

    ! Define the global grid side sizes
    integer, parameter :: global_side_size = 64
    integer(c_intptr_t) :: gns(ndims) = &
        [global_side_size, global_side_size, global_side_size]

    ! Define the global grid side lengths in X and Y
    ! The Z length instead comes from the z and z_u arrays (see below)
    real(c_float) :: lens(2) = [1.0_c_float, 1.0_c_float]

    ! Define the boundary conditions
    ! in the order of {west, east, south, north, bottom, top}
    integer :: bcs(2 * ndims) = [ &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag, &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag, &
        PoisSolver_NeumannStag, PoisSolver_NeumannStag]

    ! ns and offs receive the local grid size and offset
    ! of the local grid from the origin
    ! Use these to initialize the solver
    integer(c_intptr_t) :: local_ns(ndims), local_offs(ndims)
    integer :: ns(ndims), offs(ndims)

    ! The Z axis is nonuniform, so two arrays describe it.
    ! z_u holds the cell faces: gns(3) + 1 values, from the bottom up.
    ! z holds the cell centers: gns(3) values, one per cell.
    ! Cell k reaches from z_u(k) to z_u(k + 1), and z(k) lies between them.
    ! Both arrays describe the whole global Z axis: every rank builds the
    ! same values here.
    real(c_float), parameter :: len_z = 1.0_c_float
    real(c_float) :: dz
    integer :: nz_global
    integer :: k
    real(c_float), allocatable :: z_u(:), z(:)

    ! Grid will have two ghost cells on every side
    ! One axis will therefore have four ghost cells
    ! (two on the left, two on the right)
    integer, parameter :: ngs = 2

    ! The buffers hold the local grid and the ghost cells on both sides.
    ! There is no ngs argument on the Fortran side: poisson_solver_execute
    ! reads the ghost-cell count back out of the array bounds, so the shape
    ! alone must carry it. The local size is only known after
    ! poisson_solver_get_local_grid_info runs, so these stay allocatable.
    real(c_float), allocatable :: local_rhs(:, :, :), local_phi(:, :, :)

    type(PoissonSolver) :: solver

    call MPI_Init(ie)

    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ie)
    if (comm_size /= 4) then
        write (*, *) "This example requires exactly 4 processes"
        call MPI_Finalize(ie)
        stop 1
    end if

    ! Decomposition into a 1x2x2 process grid
    ! poisson_solver_init_mpi_grid expects nps reversed, as
    ! [processes_z, processes_y], not the [Y, Z] order the C header uses.
    ! Both entries are 2 here, so the order makes no numeric difference.
    nps = [2, 2]
    call poisson_solver_init_mpi_grid(MPI_COMM_WORLD, nps, cart_comm)

    ! Selects local device using round-robin approach
    ! Omit this if you already selected a local device
    ! (e.g. via your own mechanism)
    call poisson_solver_select_local_device(MPI_COMM_WORLD, local_device)

    call poisson_solver_get_local_grid_info(gns, cart_comm, local_ns, &
        local_offs)
    ns = int(local_ns)
    offs = int(local_offs)

    ! Even spacing keeps this example short.
    ! A real case may space the faces unevenly.
    nz_global = int(gns(3))
    dz = len_z / real(nz_global, c_float)

    allocate (z_u(0:nz_global))
    do k = 0, nz_global
        z_u(k) = real(k, c_float) * dz
    end do

    allocate (z(0:nz_global - 1))
    do k = 0, nz_global - 1
        z(k) = (z_u(k) + z_u(k + 1)) / 2
    end do

    allocate (local_rhs(1 - ngs : ns(1) + ngs, &
                        1 - ngs : ns(2) + ngs, &
                        1 - ngs : ns(3) + ngs))
    allocate (local_phi(1 - ngs : ns(1) + ngs, &
                        1 - ngs : ns(2) + ngs, &
                        1 - ngs : ns(3) + ngs))

    ! Prepare input
    ! Fill it with desired values (here, we fill 1.0)
    local_rhs = 1.0_c_float

    ! Prepare result buffer
    local_phi = 0.0_c_float

    ! Initialize the solver
    ! Use the local grid information here
    call poisson_solver_nonuniform_z_new_float(solver, ns, lens, z, z_u, &
        bcs, PoisSolver_FiniteDifference2, int(gns), offs, cart_comm)

    ! Solve the equation
    ! (local_phi and local_rhs have the same amount of ghost cells)
    call poisson_solver_nonuniform_z_execute_float(solver, local_phi, &
        local_rhs)

    ! Explicit finalization
    ! Finalize the solver before MPI_Comm_free and MPI_Finalize below
    call poisson_solver_nonuniform_z_finalize_float(solver)

    call MPI_Comm_free(cart_comm, ie)
    call MPI_Finalize(ie)

    write (*, *) "Successfully finished example distributed_nonuniform"

end program distributed_nonuniform
