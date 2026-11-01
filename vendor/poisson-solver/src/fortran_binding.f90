module PoisSolver
    use iso_c_binding
    implicit none

    type, bind(C) :: PoissonSolver
        type(c_ptr)       :: handle    = c_null_ptr
        integer(c_size_t) :: grid_size = 0
        integer(c_size_t) :: dim = 0
        integer(c_size_t) :: nx = 1, ny = 1, nz = 1
    end type PoissonSolver

    integer(c_int), parameter :: PoisSolver_Periodic       = 0
    integer(c_int), parameter :: PoisSolver_Dirichlet      = 1
    integer(c_int), parameter :: PoisSolver_Neumann        = 2
    integer(c_int), parameter :: PoisSolver_DirichletStag  = 3
    integer(c_int), parameter :: PoisSolver_NeumannStag    = 4

    integer(c_int), parameter :: PoisSolver_Spectral               = 0
    integer(c_int), parameter :: PoisSolver_FiniteDifference2      = 2
    integer(c_int), parameter :: PoisSolver_FiniteDifference4      = 4

    ! === Uniform Solvers ===
    ! == Doubles ==
    interface

        subroutine poisson_solver_new_c(solver_out, dims, ns, lens, bcs, approx, gns, offs, mpi_comm) &
                bind(C, name="PoissonSolverNewForFortran")
            import :: c_int, c_double, c_ptr, PoissonSolver
            type(PoissonSolver), intent(out)      :: solver_out
            integer(c_int),      value            :: dims
            integer(c_int),      intent(in)       :: ns(*)
            real(c_double),      intent(in)       :: lens(*)
            integer(c_int),      intent(in)       :: bcs(*)
            integer(c_int),      value            :: approx
            integer(c_int),         intent(in)       :: gns(*)
            integer(c_int),         intent(in)       :: offs(*)
            type(c_ptr),         value            :: mpi_comm
        end subroutine poisson_solver_new_c

        subroutine poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverExecute")
            import :: c_int, c_double, c_ptr, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: phi(*)
            real(c_double),      intent(in)    :: rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_execute_c

        subroutine poisson_solver_execute_gpu_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverExecuteGPU")
            import :: c_int, c_double, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: d_phi(*)
            real(c_double),      intent(in)    :: d_rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_execute_gpu_c

        subroutine poisson_solver_finalize_c(solver) &
                bind(C, name="PoissonSolverFinalize")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_finalize_c

    end interface

    ! == Floats ==
    interface

        subroutine poisson_solver_new_float_c(solver_out, dims, ns, lens, bcs, approx, gns, offs, mpi_comm) &
                bind(C, name="PoissonSolverNewFloatForFortran")
            import :: c_int, c_float, c_ptr, PoissonSolver
            type(PoissonSolver), intent(out)      :: solver_out
            integer(c_int),      value            :: dims
            integer(c_int),      intent(in)       :: ns(*)
            real(c_float),       intent(in)       :: lens(*)
            integer(c_int),      intent(in)       :: bcs(*)
            integer(c_int),      value            :: approx
            integer(c_int),         intent(in)       :: gns(*)
            integer(c_int),         intent(in)       :: offs(*)
            type(c_ptr),         value            :: mpi_comm
        end subroutine poisson_solver_new_float_c

        subroutine poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverExecuteFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: phi(*)
            real(c_float),      intent(in)    :: rhs(*)
            integer(c_int),     intent(in)    :: ngs_phi(*)
            integer(c_int),     intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_execute_float_c

        subroutine poisson_solver_execute_gpu_float_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverExecuteGPUFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: d_phi(*)
            real(c_float),      intent(in)    :: d_rhs(*)
            integer(c_int),     intent(in)    :: ngs_phi(*)
            integer(c_int),     intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_execute_gpu_float_c

        subroutine poisson_solver_finalize_float_c(solver) &
                bind(C, name="PoissonSolverFinalizeFloat")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_finalize_float_c

    end interface

    ! === Nonuniform Solvers ===
    ! == Doubles ==
    interface

         subroutine poisson_solver_nonuniform_z_new_c(solver_out, ns, lens, &
                 z, z_u, bcs, approx, gns, offs, mpi_comm) &
                bind(C, name="PoissonSolverNonuniformZNewForFortran")
             import :: c_int, c_double, c_ptr, PoissonSolver
             type(PoissonSolver), intent(out) :: solver_out
             integer(c_int),      intent(in)  :: ns(*)
             real(c_double),      intent(in)  :: lens(*)
             real(c_double), intent(in) :: z(*)
             real(c_double), intent(in) :: z_u(*)
             integer(c_int),      intent(in)  :: bcs(*)
             integer(c_int),      value       :: approx
             integer(c_int),      intent(in)  :: gns(*)
             integer(c_int),      intent(in)  :: offs(*)
             type(c_ptr),         value       :: mpi_comm
        end subroutine poisson_solver_nonuniform_z_new_c

        subroutine poisson_solver_nonuniform_z_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverNonuniformZExecute")
            import :: c_int, c_double, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: phi(*)
            real(c_double),      intent(in)    :: rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_nonuniform_z_execute_c

        subroutine poisson_solver_nonuniform_z_execute_gpu_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverNonuniformZExecuteGPU")
            import :: c_int, c_double, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: d_phi(*)
            real(c_double),      intent(in)    :: d_rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_nonuniform_z_execute_gpu_c

        subroutine poisson_solver_nonuniform_z_finalize_c(solver) &
                bind(C, name="PoissonSolverNonuniformZFinalize")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_nonuniform_z_finalize_c

    end interface

    ! == Floats ==
    interface

         subroutine poisson_solver_nonuniform_z_new_float_c(solver_out, ns, lens, &
                 z, z_u, bcs, approx, gns, offs, mpi_comm) &
                bind(C, name="PoissonSolverNonuniformZNewFloatForFortran")
             import :: c_int, c_float, c_ptr, PoissonSolver
             type(PoissonSolver), intent(out) :: solver_out
             integer(c_int),      intent(in)  :: ns(*)
             real(c_float),      intent(in)  :: lens(*)
             real(c_float), intent(in) :: z(*)
             real(c_float), intent(in) :: z_u(*)
             integer(c_int),      intent(in)  :: bcs(*)
             integer(c_int),      value       :: approx
             integer(c_int),      intent(in)  :: gns(*)
             integer(c_int),      intent(in)  :: offs(*)
             type(c_ptr),         value       :: mpi_comm
        end subroutine poisson_solver_nonuniform_z_new_float_c

        subroutine poisson_solver_nonuniform_z_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverNonuniformZExecuteFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: phi(*)
            real(c_float),      intent(in)    :: rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_nonuniform_z_execute_float_c

        subroutine poisson_solver_nonuniform_z_execute_gpu_float_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverNonuniformZExecuteGPUFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: d_phi(*)
            real(c_float),      intent(in)    :: d_rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_nonuniform_z_execute_gpu_float_c

        subroutine poisson_solver_nonuniform_z_finalize_float_c(solver) &
                bind(C, name="PoissonSolverNonuniformZFinalizeFloat")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_nonuniform_z_finalize_float_c
    end interface

    interface poisson_solver_execute
        module procedure poisson_solver_execute_1d
        module procedure poisson_solver_execute_2d
        module procedure poisson_solver_execute_3d
    end interface

    interface poisson_solver_execute_gpu
        module procedure poisson_solver_execute_1d_gpu
        module procedure poisson_solver_execute_2d_gpu
        module procedure poisson_solver_execute_3d_gpu
    end interface

    interface poisson_solver_execute_float
        module procedure poisson_solver_execute_1d_float
        module procedure poisson_solver_execute_2d_float
        module procedure poisson_solver_execute_3d_float
    end interface

    interface poisson_solver_execute_gpu_float
        module procedure poisson_solver_execute_1d_gpu_float
        module procedure poisson_solver_execute_2d_gpu_float
        module procedure poisson_solver_execute_3d_gpu_float
    end interface

#ifdef POISSON_SOLVER_USE_MPI
    ! === MPI Utils ===
    interface
        subroutine poisson_solver_init_mpi_grid_c(comm_old, nps, comm_out) &
                bind(C, name="PoissonSolverInitMPIGridForFortran")
            import :: c_int
            integer(c_int), intent(in)  :: comm_old
            integer(c_int), intent(in)  :: nps(2)
            integer(c_int), intent(out) :: comm_out
        end subroutine poisson_solver_init_mpi_grid_c

        subroutine poisson_solver_get_local_grid_info_c(gns, cart_comm, ns, offs) &
                bind(C, name="PoissonSolverGetLocalGridInfoForFortran")
            import :: c_int
            integer(c_int), intent(in)  :: gns(3)
            integer(c_int), intent(in)  :: cart_comm
            integer(c_int), intent(out) :: ns(3)
            integer(c_int), intent(out) :: offs(3)
        end subroutine poisson_solver_get_local_grid_info_c

        subroutine poisson_solver_select_local_device_c(comm, device) &
                bind(C, name="PoissonSolverSelectLocalDeviceForFortran")
            import :: c_int
            integer(c_int), intent(in)  :: comm
            integer(c_int), intent(out) :: device
        end subroutine poisson_solver_select_local_device_c
    end interface
#endif // POISSON_SOLVER_USE_MPI

contains
    ! === Uniform Solvers ===
    ! == Doubles ==

    ! Creates a new solver, writing the result into `solver`.
    !
    ! dims         : number of spatial dimensions (1, 2, or 3)
    ! ns(dims)     : grid point count per axis
    ! lens(dims)   : domain length per axis
    ! bcs(2*dims)  : boundary conditions, laid out as
    !                [west, east, south, north, bottom, top]
    ! approx       : one of PoissonSolver_Spectral,
    !                       PoissonSolver_FiniteDifference2,
    !                       PoissonSolver_FiniteDifference4
    ! gns(dims)    : global grid point count per axis
    ! offs(dims)   : offset of local data
    ! mpi_comm     : the mpi communicator
    subroutine poisson_solver_new(solver, dims, ns, lens, bcs, approx, gns, offs, mpi_comm)
        type(PoissonSolver), intent(out)          :: solver
        integer,             intent(in)           :: dims
        integer,             intent(in)           :: ns(dims)
        real(c_double),      intent(in)           :: lens(dims)
        integer,             intent(in)           :: bcs(2 * dims)
        integer,             intent(in)           :: approx
        integer,             intent(in), optional :: gns(dims)
        integer,             intent(in), optional :: offs(dims)
        integer,             intent(in), optional, target :: mpi_comm

        integer(c_int) :: local_gns(3), local_offs(3)
        type(c_ptr) :: c_comm

        if(present(gns)) local_gns(1:dims) = int(gns, c_int)
        if(present(offs)) local_offs(1:dims) = int(offs, c_int)

        c_comm = c_null_ptr
        if(present(mpi_comm)) then
            c_comm = c_loc(mpi_comm)
        end if
        
        call poisson_solver_new_c(solver, int(dims, c_int), &
            int(ns, c_int), lens, &
            int(bcs, c_int), int(approx, c_int), local_gns, local_offs, c_comm)
    end subroutine poisson_solver_new

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_1d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:)
        real(c_double),      intent(in)    :: rhs(:)
        integer(c_int) :: ngs_phi(1), ngs_rhs(1)

        ngs_phi = int((shape(phi) - [solver%nx]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx]) / 2, c_int)

        call poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_1d

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_2d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :)
        real(c_double),      intent(in)    :: rhs(:, :)
        integer(c_int) :: ngs_phi(2), ngs_rhs(2)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny]) / 2, c_int)

        call poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_2d

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_3d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :, :)
        real(c_double),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_3d

    subroutine poisson_solver_execute_1d_gpu(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: d_phi(:)
        real(c_double),      intent(in)    :: d_rhs(:)
        integer(c_int) :: ngs_phi(1), ngs_rhs(1)

        ngs_phi = int((shape(d_phi) - [solver%nx]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx]) / 2, c_int)

        call poisson_solver_execute_gpu_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_1d_gpu

    subroutine poisson_solver_execute_2d_gpu(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: d_phi(:, :)
        real(c_double),      intent(in)    :: d_rhs(:, :)
        integer(c_int) :: ngs_phi(2), ngs_rhs(2)

        ngs_phi = int((shape(d_phi) - [solver%nx, solver%ny]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx, solver%ny]) / 2, c_int)

        call poisson_solver_execute_gpu_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_2d_gpu

    subroutine poisson_solver_execute_3d_gpu(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: d_phi(:, :, :)
        real(c_double),      intent(in)    :: d_rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(d_phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_execute_gpu_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_3d_gpu

    subroutine poisson_solver_finalize(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_c(solver)
    end subroutine poisson_solver_finalize

    ! == Floats ==

    ! Creates a new solver, writing the result into `solver`.
    !
    ! dims         : number of spatial dimensions (1, 2, or 3)
    ! ns(dims)     : grid point count per axis
    ! lens(dims)   : domain length per axis
    ! bcs(2*dims)  : boundary conditions, laid out as
    !                [west, east, south, north, bottom, top]
    ! approx       : one of PoissonSolver_Spectral,
    !                       PoissonSolver_FiniteDifference2,
    !                       PoissonSolver_FiniteDifference4
    ! gns(dims)    : global grid point count per axis
    ! offs(dims)   : offset of local data
    ! mpi_comm     : the mpi communicator
    subroutine poisson_solver_new_float(solver, dims, ns, lens, bcs, approx, gns, offs, mpi_comm)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: dims
        integer,             intent(in)  :: ns(dims)
        real(c_float),      intent(in)  :: lens(dims)
        integer,             intent(in)  :: bcs(2 * dims)
        integer,             intent(in)  :: approx
        integer,             intent(in), optional :: gns(dims)
        integer,             intent(in), optional :: offs(dims)
        integer,             intent(in), optional, target :: mpi_comm

        integer(c_int) :: local_gns(3), local_offs(3)
        type(c_ptr) :: c_comm

        if(present(gns)) local_gns(1:dims) = int(gns, c_int)
        if(present(offs)) local_offs(1:dims) = int(offs, c_int)

        c_comm = c_null_ptr
        if(present(mpi_comm)) then
            c_comm = c_loc(mpi_comm)
        end if
        
        call poisson_solver_new_float_c(solver, int(dims, c_int), &
            int(ns, c_int), lens, &
            int(bcs, c_int), int(approx, c_int), local_gns, local_offs, c_comm)
    end subroutine poisson_solver_new_float

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_1d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:)
        real(c_float),      intent(in)    :: rhs(:)
        integer(c_int) :: ngs_phi(1), ngs_rhs(1)

        ngs_phi = int((shape(phi) - [solver%nx]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx]) / 2, c_int)

        call poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_1d_float

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_2d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :)
        real(c_float),      intent(in)    :: rhs(:, :)
        integer(c_int) :: ngs_phi(2), ngs_rhs(2)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny]) / 2, c_int)

        call poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_2d_float

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_3d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :, :)
        real(c_float),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_3d_float

    subroutine poisson_solver_execute_1d_gpu_float(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: d_phi(:)
        real(c_float),      intent(in)    :: d_rhs(:)
        integer(c_int) :: ngs_phi(1), ngs_rhs(1)

        ngs_phi = int((shape(d_phi) - [solver%nx]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx]) / 2, c_int)

        call poisson_solver_execute_gpu_float_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_1d_gpu_float

    subroutine poisson_solver_execute_2d_gpu_float(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: d_phi(:, :)
        real(c_float),      intent(in)    :: d_rhs(:, :)
        integer(c_int) :: ngs_phi(2), ngs_rhs(2)

        ngs_phi = int((shape(d_phi) - [solver%nx, solver%ny]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx, solver%ny]) / 2, c_int)

        call poisson_solver_execute_gpu_float_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_2d_gpu_float

    subroutine poisson_solver_execute_3d_gpu_float(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: d_phi(:, :, :)
        real(c_float),      intent(in)    :: d_rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(d_phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_execute_gpu_float_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_3d_gpu_float

    subroutine poisson_solver_finalize_float(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_float_c(solver)
    end subroutine poisson_solver_finalize_float

    ! === Nonuniform Solvers ===
    ! == Doubles ==
    subroutine poisson_solver_nonuniform_z_new(solver, ns, lens, &
            z, z_u, bcs, approx, gns, offs, mpi_comm)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: ns(3)
        real(c_double),      intent(in)  :: lens(2)
        real(c_double),      intent(in)  :: z(:)
        real(c_double),      intent(in)  :: z_u(:)
        integer,             intent(in)  :: bcs(2 * 3)
        integer,             intent(in)  :: approx
        integer,             intent(in), optional :: gns(3)
        integer,             intent(in), optional :: offs(3)
        integer,             intent(in), optional, target :: mpi_comm
        
        integer(c_int) :: local_gns(3), local_offs(3)
        type(c_ptr) :: c_comm

        if(present(gns)) local_gns(1:3) = int(gns, c_int)
        if(present(offs)) local_offs(1:3) = int(offs, c_int)

        c_comm = c_null_ptr
        if(present(mpi_comm)) then
            c_comm = c_loc(mpi_comm)
        end if

        call poisson_solver_nonuniform_z_new_c(solver, int(ns, c_int), lens, &
            z, z_u, int(bcs, c_int), int(approx, c_int), local_gns, local_offs, c_comm)
    end subroutine poisson_solver_nonuniform_z_new

    subroutine poisson_solver_nonuniform_z_execute(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :, :)
        real(c_double),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_nonuniform_z_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_nonuniform_z_execute

    subroutine poisson_solver_nonuniform_z_execute_gpu(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: d_phi(:, :, :)
        real(c_double),      intent(in)    :: d_rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(d_phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_nonuniform_z_execute_gpu_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_nonuniform_z_execute_gpu

    subroutine poisson_solver_nonuniform_z_finalize(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_nonuniform_z_finalize_c(solver)
    end subroutine poisson_solver_nonuniform_z_finalize

    ! == Floats ==
    subroutine poisson_solver_nonuniform_z_new_float(solver, ns, lens,&
            z, z_u, bcs, approx, gns, offs, mpi_comm)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: ns(3)
        real(c_float),      intent(in)  :: lens(2)
        real(c_float),      intent(in)  :: z(:)
        real(c_float),      intent(in)  :: z_u(:)
        integer,             intent(in)  :: bcs(2 * 3)
        integer,             intent(in)  :: approx
        integer,             intent(in), optional :: gns(3)
        integer,             intent(in), optional :: offs(3)
        integer,             intent(in), optional, target :: mpi_comm
        
        integer(c_int) :: local_gns(3), local_offs(3)
        type(c_ptr) :: c_comm

        if(present(gns)) local_gns(1:3) = int(gns, c_int)
        if(present(offs)) local_offs(1:3) = int(offs, c_int)

        c_comm = c_null_ptr
        if(present(mpi_comm)) then
            c_comm = c_loc(mpi_comm)
        end if

        call poisson_solver_nonuniform_z_new_float_c(solver, int(ns, c_int), lens, &
            z, z_u, int(bcs, c_int), int(approx, c_int), local_gns, local_offs, c_comm)
    end subroutine poisson_solver_nonuniform_z_new_float

    subroutine poisson_solver_nonuniform_z_execute_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :, :)
        real(c_float),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_nonuniform_z_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_nonuniform_z_execute_float

    subroutine poisson_solver_nonuniform_z_execute_gpu_float(solver, d_phi, d_rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: d_phi(:, :, :)
        real(c_float),      intent(in)    :: d_rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(d_phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(d_rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_nonuniform_z_execute_gpu_float_c(solver, d_phi, d_rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_nonuniform_z_execute_gpu_float

    subroutine poisson_solver_nonuniform_z_finalize_float(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_nonuniform_z_finalize_float_c(solver)
    end subroutine poisson_solver_nonuniform_z_finalize_float

#ifdef POISSON_SOLVER_USE_MPI
    ! === MPI Utils ===
    subroutine poisson_solver_init_mpi_grid(comm_old, nps, comm_out)
        integer, intent(in)  :: comm_old
        integer, intent(in)  :: nps(2)
        integer, intent(out) :: comm_out

        call poisson_solver_init_mpi_grid_c(comm_old, int([nps(2), nps(1)], c_int), comm_out)
    end subroutine poisson_solver_init_mpi_grid

    subroutine poisson_solver_get_local_grid_info(gns, cart_comm, ns, offs)
        integer(c_intptr_t), intent(in)  :: gns(3)
        integer, intent(in)  :: cart_comm
        integer(c_intptr_t), intent(out) :: ns(3)
        integer(c_intptr_t), intent(out) :: offs(3)
        integer(c_int) :: c_ns(3), c_offs(3)

        call poisson_solver_get_local_grid_info_c(int(gns, c_int), cart_comm, c_ns, c_offs)
        ns = int(c_ns, c_intptr_t)
        offs = int(c_offs, c_intptr_t)
    end subroutine poisson_solver_get_local_grid_info

    subroutine poisson_solver_select_local_device(comm, device_out)
        integer, intent(in)  :: comm
        integer, intent(out) :: device_out

        call poisson_solver_select_local_device_c(comm, device_out)
    end subroutine poisson_solver_select_local_device
#endif // POISSON_SOLVER_USE_MPI

end module PoisSolver
