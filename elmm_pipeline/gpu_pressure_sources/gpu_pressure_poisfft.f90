! Qualified metadata adapter for the pinned, local, uniform PoisFFT DP solver.
! This is an ordinary typed Fortran interface: D is neither copied nor turned
! into an interoperable pointer. Its original constructor must already have
! run at the original reached CALL. This query grants no source/lifetime proof.
module elmm_gpu_pressure_poisfft
  use, intrinsic :: iso_c_binding
  use PoisFFT, only: PoisFFT_Solver3D_DP
  use elmm_gpu_pressure, only: elmm_gpu_pressure_config_v1
  use fort_scoped_memory, only: FORT_SCOPE_OK, FORT_SCOPE_ARGUMENT, FORT_SCOPE_BOUNDARY
  implicit none
  private
  public :: elmm_gpu_pressure_configure_poisfft_v1
contains
  integer(c_int) function elmm_gpu_pressure_configure_poisfft_v1(config, D) result(status)
    type(elmm_gpu_pressure_config_v1), intent(inout) :: config
    type(PoisFFT_Solver3D_DP), intent(in) :: D
    type(elmm_gpu_pressure_config_v1) :: proposed
    integer(c_int64_t) :: bits(3)
    integer :: axis

    status = FORT_SCOPE_ARGUMENT
    if (config%version /= 1 .or. config%reserved /= 0) return
    if (config%solver%device_budget_bytes <= 0) return
    if (any(D%nxyz < 2) .or. any(D%nxyz > huge(0_c_int32_t)-4)) return
    if (any(D%nxyz /= [D%nx, D%ny, D%nz])) return
    if (D%approximation /= 2 .and. D%approximation /= 4) return
    ! Check the binary64 length representation without floating arithmetic:
    ! malformed signaling-NaN metadata must not change host IEEE flags.
    if (storage_size(D%Lx) /= 64 .or. storage_size(0_c_double) /= 64) return
    bits = [transfer(D%Lx, 0_c_int64_t), transfer(D%Ly, 0_c_int64_t), transfer(D%Lz, 0_c_int64_t)]
    do axis = 1, 3
      if (btest(bits(axis), 63)) return
      if (iand(bits(axis), int(z'7fffffffffffffff', c_int64_t)) == 0) return
      if (iand(bits(axis), int(z'7ff0000000000000', c_int64_t)) == int(z'7ff0000000000000', c_int64_t)) return
    end do

    status = FORT_SCOPE_BOUNDARY
    if (D%mpi%comm /= -1) return
    if (any([D%offx,D%offy,D%offz] /= 0)) return
    if (any([D%gnx,D%gny,D%gnz] /= D%nxyz)) return
    ! These three uniform boundary families have pinned differential and
    ! CPU scratch-reinitialization qualifications. Other families stay native.
    if (.not. (all(D%BCs == 0) .or. all(D%BCs == 4) .or. &
               (all(D%BCs(1:4) == 0) .and. all(D%BCs(5:6) == 4)))) return

    proposed = config
    proposed%solver%version = 1
    proposed%solver%dimensions = 3
    proposed%solver%flags = 0
    proposed%solver%sizes = D%nxyz
    proposed%solver%lengths = [D%Lx,D%Ly,D%Lz]
    proposed%solver%boundary_conditions = D%BCs
    proposed%solver%approximation = D%approximation
    config = proposed
    status = FORT_SCOPE_OK
  end function
end module
