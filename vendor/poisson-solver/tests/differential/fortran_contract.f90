program contract
    use, intrinsic :: iso_c_binding
    use poisson_solver_device
    implicit none
    type(poisson_solver_config_v1) :: config
    type(poisson_solver_workspace_v1) :: workspace
    type(c_ptr) :: handle
    integer(c_int) :: status, started
    interface
        function verify_fortran_layout(c, bytes) bind(C) result(status)
            import :: c_int, c_size_t, poisson_solver_config_v1
            type(poisson_solver_config_v1), intent(in) :: c
            integer(c_size_t), value :: bytes
            integer(c_int) :: status
        end function
    end interface
    config%sizes=[2,3,4]
    config%lengths=[.25_c_double,.5_c_double,.75_c_double]
    config%boundary_conditions=[0,1,2,3,4,5]
    config%approximation=4
    config%flags=7
    config%device_budget_bytes=int(z'112233445566',c_int64_t)
    if(verify_fortran_layout(config,c_sizeof(config))/=0) stop 7
    config%flags=0
    config%version=0
    status=poisson_solver_create_v1(config,handle)
    if(status/=poisson_solver_argument .or. c_associated(handle)) stop 1
    config%version=1
    config%flags=1 ! Checks a field after all fixed arrays in the C layout.
    status=poisson_solver_create_v1(config,handle)
    if(status/=poisson_solver_argument .or. c_associated(handle)) stop 2
    config%flags=0
    config%dimensions=4
    status=poisson_solver_create_v1(config,handle)
    if(status/=poisson_solver_unsupported .or. c_associated(handle)) stop 3
    status=poisson_solver_workspace_query_v1(handle,workspace)
    if(status/=poisson_solver_argument) stop 4
    started=99
    status=poisson_solver_execute_device_v1(handle,c_null_ptr,0_c_size_t,c_null_ptr,0_c_size_t, &
        [2_c_int32_t,2_c_int32_t,2_c_int32_t],[1_c_int32_t,1_c_int32_t,1_c_int32_t],c_null_ptr,started)
    if(status/=poisson_solver_argument .or. started/=0) stop 5
    status=poisson_solver_destroy_v1(handle)
    if(status/=poisson_solver_ok) stop 6
    print *, 'PASS Fortran c_ptr/status bindings without CUDA initialization'
end program
