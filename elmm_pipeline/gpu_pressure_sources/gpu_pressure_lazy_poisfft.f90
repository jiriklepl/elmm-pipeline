! Explicit opt-in solver implementation. This is not the original PoisFFT
! receiver, and requires its own source/numerical qualification. Construction
! records the local operator only. A CPU FFT plan is created only if the
! original Execute path is actually reached after a safe device rejection.
! GPU plans remain owned by the compiler's checked outer domain/provider slot.
module elmm_gpu_pressure_lazy_poisfft
  use, intrinsic :: iso_c_binding
  use PoisFFT, only: native_solver => PoisFFT_Solver3D_DP, native_execute => Execute, &
                    native_finalize => Finalize, PoisFFT_FiniteDifference2, PoisFFT_FiniteDifference4
  use elmm_gpu_pressure, only: elmm_gpu_pressure_config_v1
  use fort_scoped_memory, only: FORT_SCOPE_OK, FORT_SCOPE_ARGUMENT, FORT_SCOPE_BOUNDARY
  implicit none
  private
  public :: PoisFFT_Solver3D_DP, Execute, Finalize, elmm_gpu_pressure_configure_lazy_v1
  public :: PoisFFT_FiniteDifference2, PoisFFT_FiniteDifference4
  public :: elmm_lazy_cpu_initializations

  type :: PoisFFT_Solver3D_DP
    private
    integer :: nxyz(3)=0, bcs(6)=0, approximation=0
    real(c_double) :: lengths(3)=0
    logical :: configured=.false.
    type(native_solver), allocatable :: cpu
    integer :: cpu_initializations=0
  end type

  interface PoisFFT_Solver3D_DP
    module procedure lazy_new
  end interface
  interface Execute
    module procedure lazy_execute
  end interface
  interface Finalize
    module procedure lazy_finalize
  end interface
contains
  pure function lazy_new(nxyz, Lxyz, bcs, approximation) result(D)
    integer, intent(in) :: nxyz(3), bcs(6), approximation
    real(c_double), intent(in) :: Lxyz(3)
    type(PoisFFT_Solver3D_DP) :: D
    D%nxyz = nxyz
    D%lengths = Lxyz
    D%bcs = bcs
    D%approximation = approximation
    D%configured = .true.
  end function

  subroutine lazy_execute(D, Phi, RHS)
    type(PoisFFT_Solver3D_DP), intent(inout) :: D
    real(c_double), intent(out) :: Phi(:,:,:)
    real(c_double), intent(in) :: RHS(:,:,:)
    if (.not. D%configured) error stop 'unconfigured lazy PoisFFT receiver'
    if (.not. allocated(D%cpu)) then
      allocate(D%cpu)
      D%cpu = native_solver(D%nxyz, D%lengths, D%bcs, D%approximation)
      D%cpu_initializations = D%cpu_initializations + 1
    endif
    call native_execute(D%cpu, Phi, RHS)
  end subroutine

  subroutine lazy_finalize(D)
    type(PoisFFT_Solver3D_DP), intent(inout) :: D
    if (allocated(D%cpu)) then
      call native_finalize(D%cpu)
      deallocate(D%cpu)
    endif
  end subroutine

  pure integer function elmm_lazy_cpu_initializations(D) result(count)
    type(PoisFFT_Solver3D_DP), intent(in) :: D
    count = D%cpu_initializations
  end function

  integer(c_int) function elmm_gpu_pressure_configure_lazy_v1(config, D) result(status)
    type(elmm_gpu_pressure_config_v1), intent(inout) :: config
    type(PoisFFT_Solver3D_DP), intent(in) :: D
    type(elmm_gpu_pressure_config_v1) :: proposed
    integer(c_int64_t) :: bits(3)
    integer :: axis

    status = FORT_SCOPE_ARGUMENT
    if (.not. D%configured) return
    if (config%version /= 1 .or. config%reserved /= 0) return
    if (config%solver%device_budget_bytes <= 0) return
    if (any(D%nxyz < 2) .or. any(D%nxyz > huge(0_c_int32_t)-4)) return
    if (D%approximation /= 2 .and. D%approximation /= 4) return
    if (storage_size(D%lengths) /= 64 .or. storage_size(0_c_double) /= 64) return
    bits = transfer(D%lengths, bits)
    do axis = 1, 3
      if (btest(bits(axis), 63)) return
      if (iand(bits(axis), int(z'7fffffffffffffff', c_int64_t)) == 0) return
      if (iand(bits(axis), int(z'7ff0000000000000', c_int64_t)) == int(z'7ff0000000000000', c_int64_t)) return
    enddo
    status = FORT_SCOPE_BOUNDARY
    if (.not. (all(D%bcs == 0) .or. all(D%bcs == 4) .or. &
               (all(D%bcs(1:4) == 0) .and. all(D%bcs(5:6) == 4)))) return
    proposed = config
    proposed%solver%version = 1
    proposed%solver%dimensions = 3
    proposed%solver%flags = 0
    proposed%solver%sizes = D%nxyz
    proposed%solver%lengths = D%lengths
    proposed%solver%boundary_conditions = D%bcs
    proposed%solver%approximation = D%approximation
    config = proposed
    status = FORT_SCOPE_OK
  end function
end module
