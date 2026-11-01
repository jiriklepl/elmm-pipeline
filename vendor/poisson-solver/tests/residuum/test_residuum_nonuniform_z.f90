! Originally from PoisFFT by Vladimir Fuka.
! Modified by Matous Zatloukal.
! Changes: Swapped PoisFFT API calls with poisson-solver API calls
! for testing purposes and added poisson-solver GPU API tests.

module Kinds
  use iso_c_binding
  use iso_fortran_env

  integer,parameter :: rp = c_double
  integer,parameter :: cp = c_double_complex
  integer,parameter :: size_kind = c_size_t
end module

module Globals
  use Kinds
  
  implicit none

  real(rp), parameter :: pi = 3.141592653589793238462_rp!pi = 4*atan(1._rp)!
  real(rp), parameter :: Lx = 2*pi, Ly = Lx*1.1_rp, Lz = Lx/1.1_rp
  integer :: nx = 21, ny = 32, nz = 25
  real(rp) :: dx,dy,dz
  real(rp), dimension(:,:,:), allocatable, target :: Phi3D, RHS3D
  real(rp), dimension(:,:), allocatable, target :: Phi2D, RHS2D
  real(rp), dimension(:), allocatable, target :: Phi1D, RHS1D

  real(rp), dimension(:), allocatable :: z(:), z_u(:)
end module



module Residues
  use Kinds
  use Globals
  use PoisSolver
    
  implicit none
  
  integer, parameter :: We = 1,Ea = 2,So = 3,No = 4,Bo = 5,To = 6

contains

  subroutine BCond(Phi)
    real(rp), intent(inout) :: Phi(0:,0:,0:)
    Phi(0,:,:) = Phi(nx,:,:)
    Phi(nx+1,:,:) = Phi(1,:,:)
    Phi(:,0,:) = Phi(:,ny,:)
    Phi(:,ny+1,:) = Phi(:,1,:)
    Phi(:,:,0) = Phi(:,:,nz)
    Phi(:,:,nz+1) = Phi(:,:,1)
  end subroutine


  subroutine Res3D(Phi,RHS,&
                   Aw,Ae,As,An,z,z_u,&
                   Btype,R)
    !2nd order finite difference residuum

    real(rp), intent(inout) :: Phi(0:,0:,0:)
    real(rp), intent(in) :: RHS(:,:,:)
    real(rp),intent(in) :: Aw,Ae
    real(rp),intent(in) :: As,An
    real(rp),intent(in) :: z(0:),z_u(-1:)
    integer,intent(in) :: Btype(6)
    real(rp),intent(out) :: R
    integer i,j,k
    real(rp) :: p,Ap

    call BCond(Phi)
    
    R = 0

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          p = 0
          Ap = 0          
          if (i>1) then
                    p = p + Phi(i-1,j,k)*Aw
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_Periodic) then
                    p = p + Phi(nx,j,k)*Aw
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_DirichletStag) then
                    p = p - Phi(1,j,k)*Aw
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_Dirichlet) then
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_Neumann) then
                    p = p + Phi(2,j,k)*Aw
                    Ap = Ap + Aw
          end if
          if (i<nx) then
                    p = p + Phi(i+1,j,k)*Ae
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_Periodic) then
                    p = p + Phi(1,j,k)*Ae
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_DirichletStag) then
                    p = p - Phi(nx,j,k)*Ae
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_Dirichlet) then
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_Neumann) then
                    p = p + Phi(nx-1,j,k)*Ae
                    Ap = Ap + Ae
          end if
          if (j>1) then
                    p = p + Phi(i,j-1,k)*As
                    Ap = Ap + As
          elseif (Btype(So)==PoisSolver_Periodic) then
                    p = p + Phi(i,ny,k)*As
                    Ap = Ap + As
           elseif (Btype(So)==PoisSolver_DirichletStag) then
                    p = p - Phi(i,1,k)*As
                    Ap = Ap + As
          elseif (Btype(So)==PoisSolver_Dirichlet) then
                    Ap = Ap + As
          elseif (Btype(So)==PoisSolver_Neumann) then
                    p = p + Phi(i,2,k)*As
                    Ap = Ap + As
          end if
          if (j<ny) then
                    p = p + Phi(i,j+1,k)*An
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_Periodic) then
                    p = p + Phi(i,1,k)*An
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_DirichletStag) then
                    p = p - Phi(i,ny,k)*An
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_Dirichlet) then
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_Neumann) then
                    p = p + Phi(i,ny-1,k)*An
                    Ap = Ap + An
          end if
          if (k>1) then
                    p = p + Phi(i,j,k-1)/(z(k) - z(k-1))/(z_u(k)-z_u(k-1))
                    Ap = Ap + 1._rp/(z(k) - z(k-1))/(z_u(k)-z_u(k-1))
          end if
          if (k<nz) then
                    p = p + Phi(i,j,k+1)/(z(k+1) - z(k))/(z_u(k)-z_u(k-1))
                    Ap = Ap + 1._rp/(z(k+1) - z(k))/(z_u(k)-z_u(k-1))
          end if

          p = p - RHS(i,j,k)
          p = abs(-p +Ap*Phi(i,j,k))
          R = max(R,abs(p))
        end do
      end do
    end do

  end subroutine Res3D
end module










module Tests
  use Kinds
  use Globals
  use Residues
  use PoisSolver
  
  implicit none
  
  interface
    function fortran_cuda_malloc(bytes) bind(C, name="fortran_cuda_malloc")
      import :: c_size_t, c_ptr
      integer(c_size_t), value :: bytes
      type(c_ptr) :: fortran_cuda_malloc
    end function fortran_cuda_malloc

    subroutine fortran_cuda_free(ptr) bind(C, name="fortran_cuda_free")
      import :: c_ptr
      type(c_ptr), value :: ptr
    end subroutine fortran_cuda_free

    subroutine fortran_cuda_memcpy_h2d(dst, src, bytes) bind(C, name="fortran_cuda_memcpy_h2d")
      import :: c_size_t, c_ptr
      type(c_ptr), value :: dst
      type(c_ptr), value :: src
      integer(c_size_t), value :: bytes
    end subroutine fortran_cuda_memcpy_h2d

    subroutine fortran_cuda_memcpy_d2h(dst, src, bytes) bind(C, name="fortran_cuda_memcpy_d2h")
      import :: c_size_t, c_ptr
      type(c_ptr), value :: dst
      type(c_ptr), value :: src
      integer(c_size_t), value :: bytes
    end subroutine fortran_cuda_memcpy_d2h
  end interface

  character(20), parameter :: char_BCs(0:4) = ["Periodic            ", &
                                               "Dirichlet regular   ", &
                                               "Neumann regular     ", &
                                               "Dirichlet staggered ", &
                                               "Neumann staggered   "]
  
contains

  
  
  
  subroutine Test3D(BCs)
    integer, intent(in) :: BCs(6)
    integer :: i

    write(*,*) "----"
    write(*,'(1x,"3D  |",*(a,"|"))') (trim(char_BCs(BCs(i))), i=1,size(BCs))

    call TestFD2_3D(BCs)
    write(*,*)
  end subroutine

  
  subroutine TestFD2_3D(BCs)
    integer, intent(in) :: BCs(6)
    real(rp) :: R
    
    call RunFD2_3D(BCs, .false.)
    
    call Res3D(Phi3D, RHS3D, &
               dx**(-2), dx**(-2), dy**(-2), dy**(-2), z, z_u, &
               BCs, R)
    if (R < int(nx, int64) * int(ny, int64) * int(nz, int64) * 1000 * epsilon(1._rp)) then
      write(*,*) "FD2 CPU API OK", R
    else
      write(*,*) "FD2 CPU API FAIL"
      write(*,*) "FD2 CPU API residuum:",R
    end if
    call RunFD2_3D(BCs, .true.)

    call Res3D(Phi3D, RHS3D, &
               dx**(-2), dx**(-2), dy**(-2), dy**(-2), z, z_u, &
               BCs, R)

    if (R < int(nx, int64) * int(ny, int64) * int(nz, int64) * 100 * epsilon(1._rp)) then
      write(*,*) "FD2 GPU API OK", R
    else
      write(*,*) "FD2 GPU API FAIL"
      write(*,*) "FD2 GPU API residuum:",R
    end if
  end subroutine
  
  subroutine RunFD2_3D(BCs, use_gpu)
    type(PoissonSolver) :: Solver
    integer, intent(in) :: BCs(6)
    logical, intent(in), optional :: use_gpu
    logical :: gpu
    type(c_ptr) :: d_phi_ptr, d_rhs_ptr
    real(rp), pointer :: d_phi(:,:,:), d_rhs(:,:,:)
    integer(c_size_t) :: bytes_phi, bytes_rhs
  
    ! Pass lengths [Lx, Ly] as expected by the 2D xy-slice logic
    call poisson_solver_nonuniform_z_new(Solver, [nx,ny,nz], [Lx,Ly], z(1:nz), z_u(0:nz), BCs, PoisSolver_FiniteDifference2)

    ! The wrapper natively infers ngs_phi and ngs_rhs using shape(Phi3D) and shape(RHS3D)
    gpu = .false.
    if (present(use_gpu)) gpu = use_gpu
    if (gpu) then
        bytes_phi = size(Phi3D, kind=c_size_t) * c_sizeof(Phi3D(0,0,0))
        bytes_rhs = size(RHS3D, kind=c_size_t) * c_sizeof(RHS3D(1,1,1))
        d_phi_ptr = fortran_cuda_malloc(bytes_phi)
        d_rhs_ptr = fortran_cuda_malloc(bytes_rhs)
        call fortran_cuda_memcpy_h2d(d_rhs_ptr, c_loc(RHS3D(1,1,1)), bytes_rhs)
        call c_f_pointer(d_phi_ptr, d_phi, shape(Phi3D))
        call c_f_pointer(d_rhs_ptr, d_rhs, shape(RHS3D))

        call poisson_solver_nonuniform_z_execute_gpu(Solver, d_phi, d_rhs)

        call fortran_cuda_memcpy_d2h(c_loc(Phi3D(0,0,0)), d_phi_ptr, bytes_phi)
        call fortran_cuda_free(d_phi_ptr)
        call fortran_cuda_free(d_rhs_ptr)
    else
        call poisson_solver_nonuniform_z_execute(Solver, Phi3D, RHS3D)
    end if

    call poisson_solver_nonuniform_z_finalize(Solver)

  end subroutine
  

  
end module
















program testpoisson
  use Kinds
  use Globals
  use Residues
  use Tests
  use PoisSolver

  
  implicit none

  integer i,j,k,niters
  real(rp) :: xp, yp, zp
  character(len = 12) :: arg
  real(RP) :: avg, p
  
  
  call random_seed(size=j)
  call random_seed(put=[(i+1,i=1,j)])
  
  if (command_argument_count()>=3) then
    call get_command_argument(1,value = arg)
    read(arg,'(i12)') nx
    call get_command_argument(2,value = arg)
    read(arg,'(i12)') ny
    call get_command_argument(3,value = arg)
    read(arg,'(i12)') nz
  end if
  
  write(*,*) "nx",nx
  write(*,*) "ny",ny
  write(*,*) "nz",nz
  
  dx = Lx/nx;dy = Ly/ny;dz = Lz/nz
  
  allocate(z(0:nz+1))
  allocate(z_u(-1:nz+1))
  z_u(0:nz) = [(Lz/nz*i, i = 0, nz)]
  do i = 1, nz-1
    call random_number(p)
    p = (p - 0.5_rp)*0.5
     z_u(i) = z_u(i) + min(abs(z_u(i+1)-z_u(i)),abs(z_u(i)-z_u(i-1)))*p
  end do
  z_u(-1) = z_u(0) - (z_u(1)-z_u(0))
  z_u(nz+1) = z_u(nz) + (z_u(nz)-z_u(nz-1))
  do i = 0, nz+1
    z(i) = (z_u(i)+z_u(i-1))/2
  end do

  
  allocate(RHS3D(nx,ny,nz))
  allocate(Phi3D(0:nx+1,0:ny+1,0:nz+1))


  avg = 0
  do k = 1,nz
   do j = 1,ny
    do i = 1,nx
     xp = (i-1._rp/2)*dx
     yp = (j-1._rp/2)*dy
     zp = z(k)
     call random_number(RHS3D(i,j,k))
     avg = avg + RHS3D(i,j,k)*dx*dy*(z_u(k)-z_u(k-1))
     call random_number(Phi3D(i,j,k))
    end do
   end do
  end do

  avg = avg / (Lx*Ly*Lz)
  RHS3D = RHS3D - avg


  dx = Lx / nx
  dy = Ly / ny
  call Test3D([(PoisSolver_Periodic, i = 1,4),(PoisSolver_NeumannStag, i = 5,6)])

  call Test3D([(PoisSolver_NeumannStag, i = 1,6)])


end program testpoisson
