! Independent reference adapter to the pinned, unmodified CPU PoisFFT module.
subroutine reference_solve(ns, lens, bcs, approximation, phi, rhs) bind(C)
    use, intrinsic :: iso_c_binding
    use PoisFFT, only: PoisFFT_Solver3D_DP, Execute, Finalize
    implicit none
    integer(c_int), intent(in) :: ns(3), bcs(6)
    real(c_double), intent(in) :: lens(3)
    integer(c_int), value :: approximation
    type(c_ptr), value :: phi, rhs
    real(c_double), pointer :: p(:,:,:), r(:,:,:)
    type(PoisFFT_Solver3D_DP) :: solver
    call c_f_pointer(phi,p,ns+4)
    call c_f_pointer(rhs,r,ns+2)
    solver=PoisFFT_Solver3D_DP(ns,lens,bcs,approximation)
    call Execute(solver,p,r)
    call Finalize(solver)
end subroutine
