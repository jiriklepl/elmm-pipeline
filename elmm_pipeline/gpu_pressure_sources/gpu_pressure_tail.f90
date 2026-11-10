module elmm_gpu_pressure_tail
  use, intrinsic :: iso_c_binding
  use fort_scoped_memory, only: fort_scope_external_report_v1
  implicit none
  private
  public :: elmm_gpu_pressure_tail_v1
  interface
    function elmm_gpu_pressure_tail_v1(context, pressure, sizes, bcs, reference, report) bind(C) result(status)
      import :: c_int, c_int64_t, c_int32_t, fort_scope_external_report_v1
      integer(c_int64_t), value :: context, pressure
      integer(c_int32_t), intent(in) :: sizes(3), bcs(6)
      integer(c_int), value :: reference
      type(fort_scope_external_report_v1), intent(out) :: report
      integer(c_int) :: status
    end function
  end interface
end module
