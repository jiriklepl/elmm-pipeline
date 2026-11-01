! Status-returning device-pointer API. These are C pointers, never Fortran
! array descriptors or host association to device memory. Completion belongs
! to the supplied stream; the caller retains its allocations until completion.
module poisson_solver_device
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), parameter :: poisson_solver_ok=0, poisson_solver_argument=1, &
        poisson_solver_resource=2, poisson_solver_execution=3, &
        poisson_solver_unsupported=4, poisson_solver_poisoned=5
    type, bind(C) :: poisson_solver_config_v1
        integer(c_int32_t) :: version=1, dimensions=3
        integer(c_int32_t) :: sizes(3)=0
        real(c_double) :: lengths(3)=0
        integer(c_int32_t) :: boundary_conditions(6)=0
        integer(c_int32_t) :: approximation=2, flags=0
        integer(c_int64_t) :: device_budget_bytes=0
    end type
    type, bind(C) :: poisson_solver_workspace_v1
        integer(c_int32_t) :: version=0, flags=0
        integer(c_int64_t) :: persistent_device_bytes=0, peak_device_bytes=0, execution_extra_bytes=0
    end type
    interface
        function poisson_solver_create_v1(config, handle) bind(C) result(status)
            import :: c_ptr, c_int, poisson_solver_config_v1
            type(poisson_solver_config_v1), intent(in) :: config
            type(c_ptr), intent(out) :: handle
            integer(c_int) :: status
        end function
        function poisson_solver_workspace_query_v1(handle, info) bind(C) result(status)
            import :: c_ptr, c_int, poisson_solver_workspace_v1
            type(c_ptr), value :: handle
            type(poisson_solver_workspace_v1), intent(out) :: info
            integer(c_int) :: status
        end function
        function poisson_solver_execute_device_v1(handle, phi, phi_bytes, rhs, rhs_bytes, &
                ghost_phi, ghost_rhs, stream, started) bind(C) result(status)
            import :: c_ptr, c_size_t, c_int, c_int32_t
            type(c_ptr), value :: handle, phi, rhs, stream
            integer(c_size_t), value :: phi_bytes, rhs_bytes
            integer(c_int32_t), intent(in) :: ghost_phi(3), ghost_rhs(3)
            integer(c_int), intent(out) :: started
            integer(c_int) :: status
        end function
        function poisson_solver_destroy_v1(handle) bind(C) result(status)
            import :: c_ptr, c_int
            type(c_ptr), intent(inout) :: handle
            integer(c_int) :: status
        end function
        function poisson_solver_last_error_v1() bind(C) result(message)
            import :: c_ptr
            type(c_ptr) :: message
        end function
    end interface
end module
