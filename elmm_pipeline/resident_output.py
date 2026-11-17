"""Bounded immutable VTK snapshots using the compiler's output queue.

Only initialized VTK domains whose Fill reads frozen primary-channel geometry
and the five snapshotted fields are eligible. Other output retains the original
coherent host path, after joining earlier snapshots. No application arithmetic
or serializer is replaced. The worker owns a deep copy of each frame object.
"""
from .ast_rewrites import (
    find_call_span,
    find_once,
    find_routine_span,
    replace_once,
)


DECLARATIONS = """
  type resident_frame_object
    class(TFrameBase), allocatable :: frame
  end type
  type resident_frame_slot
    type(resident_frame_object), allocatable :: frames(:)
    integer :: count=0, shapes(3,5)=0
  end type
  type(resident_frame_slot), save :: resident_frames(2)
  integer(c_int32_t), target, save :: resident_slot_ids(2)=[1,2]
  type(c_ptr), save :: resident_output_queue=c_null_ptr
  integer(c_size_t), save :: resident_output_bytes=0
  integer(c_int32_t), save :: resident_pending_slot=0
  integer, save :: resident_batch_capacity=1
  type(fort_scope_external_binding_v1), target, save :: resident_pending_bindings(5)
  integer(c_size_t), target, save :: resident_pending_bytes(5)
"""

PROCEDURES = """
  subroutine BeginResidentFrameBatch(capacity)
    integer, intent(in) :: capacity
    if (resident_pending_slot/=0) error stop 'nested snapshot batch'
    resident_batch_capacity=max(1,capacity)
  end subroutine

  subroutine FlushResidentFrameBatch()
    use fort_resident_snapshots
    use elmm_resident_pressure_state, only: resident_flow, pressure_require
    if (resident_pending_slot==0) return
    call pressure_require(fort_snapshot_submit_v1(resident_output_queue,resident_pending_slot,resident_flow%context, &
      c_loc(resident_pending_bindings),c_loc(resident_pending_bytes),5_c_size_t, &
      c_funloc(WriteResidentFrame),c_loc(resident_slot_ids(resident_pending_slot))))
    resident_pending_slot=0
  end subroutine
  subroutine WaitResidentOutput()
    use fort_resident_snapshots
    use elmm_resident_pressure_state, only: pressure_require
    call FlushResidentFrameBatch()
    if (c_associated(resident_output_queue)) &
      call pressure_require(fort_snapshot_wait_v1(resident_output_queue))
  end subroutine

  subroutine FinalizeResidentOutput()
    use fort_resident_snapshots
    use elmm_resident_pressure_state, only: pressure_require
    integer :: i
    call FlushResidentFrameBatch()
    if (.not.c_associated(resident_output_queue)) return
    call pressure_require(fort_snapshot_destroy_v1(resident_output_queue))
    resident_output_queue=c_null_ptr
    resident_output_bytes=0
    do i=1,2
      if (allocated(resident_frames(i)%frames)) deallocate(resident_frames(i)%frames)
    enddo
  end subroutine

  subroutine QueueResidentFrame(D,U,V,W,Pr,Viscosity)
    use fort_scoped_memory
    use fort_resident_snapshots
    use elmm_resident_pressure_state, only: resident_flow, pressure_require
    class(TFrameBase), intent(in) :: D
    real(knd), contiguous, target, intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    real(knd), contiguous, target, intent(in) :: Pr(-1:,-1:,-1:),Viscosity(-1:,-1:,-1:)
    integer(c_size_t) :: bytes(5)
    integer(c_size_t) :: total
    integer(c_int32_t) :: slot
    integer :: i
    bytes=int(storage_size(0._knd)/8,c_size_t)*[size(U,kind=c_size_t),size(V,kind=c_size_t), &
                     size(W,kind=c_size_t),size(Pr,kind=c_size_t),size(Viscosity,kind=c_size_t)]
    total=sum(bytes)
    if (.not.c_associated(resident_output_queue)) then
      call pressure_require(fort_snapshot_create_v1(resident_flow%domain,total,268435456_c_size_t,resident_output_queue))
      resident_output_bytes=total
    endif
    if (total/=resident_output_bytes) error stop 'snapshot allocation epoch changed'
    if (resident_pending_slot==0) then
    call pressure_require(fort_snapshot_acquire_v1(resident_output_queue,slot))
    resident_pending_slot=slot
    if (allocated(resident_frames(slot)%frames)) deallocate(resident_frames(slot)%frames)
    allocate(resident_frames(slot)%frames(resident_batch_capacity))
    resident_frames(slot)%count=0
    resident_pending_bytes=bytes
    resident_frames(slot)%shapes(:,1)=shape(U)
    resident_frames(slot)%shapes(:,2)=shape(V)
    resident_frames(slot)%shapes(:,3)=shape(W)
    resident_frames(slot)%shapes(:,4)=shape(Pr)
    resident_frames(slot)%shapes(:,5)=shape(Viscosity)
    resident_pending_bindings(1)%buffer=resident_flow%bind_real(U,lbound(U),.true.,origin='frame_snapshot::u')
    resident_pending_bindings(2)%buffer=resident_flow%bind_real(V,lbound(V),.true.,origin='frame_snapshot::v')
    resident_pending_bindings(3)%buffer=resident_flow%bind_real(W,lbound(W),.true.,origin='frame_snapshot::w')
    resident_pending_bindings(4)%buffer=resident_flow%bind_real(Pr,lbound(Pr),.true.,origin='frame_snapshot::pr')
    resident_pending_bindings(5)%buffer=resident_flow%bind_real(Viscosity,lbound(Viscosity),.true., &
                                  origin='frame_snapshot::viscosity')
    do i=1,5
      resident_pending_bindings(i)%element_type=merge(FORT_SCOPE_REAL64,FORT_SCOPE_REAL32,storage_size(0._knd)==64)
      resident_pending_bindings(i)%rank=3
      resident_pending_bindings(i)%element_bytes=int(storage_size(0._knd)/8,c_size_t)
      resident_pending_bindings(i)%access%flags=FORT_SCOPE_READ_ALL
    enddo
    endif
    slot=resident_pending_slot
    resident_frames(slot)%count=resident_frames(slot)%count+1
    i=resident_frames(slot)%count
    if (i>size(resident_frames(slot)%frames)) error stop 'snapshot batch capacity exceeded'
    allocate(resident_frames(slot)%frames(i)%frame,source=D)
  end subroutine

  recursive function WriteResidentFrame(user,payload,bytes) bind(C) result(status)
    use fort_scoped_memory, only: FORT_SCOPE_OK,FORT_SCOPE_ARGUMENT
    type(c_ptr), value :: user,payload
    integer(c_size_t), value :: bytes
    integer(c_int) :: status
    integer(c_int32_t), pointer :: slot
    real(knd), pointer :: flat(:),U(:,:,:),V(:,:,:),W(:,:,:),Pr(:,:,:),Viscosity(:,:,:)
    real(knd) :: empty3(0,0,0),empty4(0,0,0,0)
    integer(c_size_t) :: first,last
    integer :: n(3,5),i,j
    call c_f_pointer(user,slot)
    status=FORT_SCOPE_ARGUMENT
    if (slot<1 .or. slot>2) return
    n=resident_frames(slot)%shapes
    if (bytes/=int(storage_size(0._knd)/8,c_size_t)*sum(product(int(n,c_size_t),dim=1))) return
    call c_f_pointer(payload,flat,[bytes/int(storage_size(0._knd)/8,c_size_t)])
    first=1
    do i=1,5
      last=first+product(int(n(:,i),c_size_t))-1
      select case(i)
      case(1)
        U(1:n(1,i),1:n(2,i),1:n(3,i))=>flat(first:last)
      case(2)
        V(1:n(1,i),1:n(2,i),1:n(3,i))=>flat(first:last)
      case(3)
        W(1:n(1,i),1:n(2,i),1:n(3,i))=>flat(first:last)
      case(4)
        Pr(1:n(1,i),1:n(2,i),1:n(3,i))=>flat(first:last)
      case(5)
        Viscosity(1:n(1,i),1:n(2,i),1:n(3,i))=>flat(first:last)
      end select
      first=last+1
    enddo
    do j=1,resident_frames(slot)%count
      call resident_frames(slot)%frames(j)%frame%Fill(U,V,W,Pr,Viscosity,empty3,empty3,empty4)
      call resident_frames(slot)%frames(j)%frame%DoSave()
      if (resident_frames(slot)%frames(j)%frame%in_progress) call resident_frames(slot)%frames(j)%frame%Wait()
    enddo
    status=FORT_SCOPE_OK
  end function
"""


def asynchronous_frames(common, vtk, main):
    """Transform the checked source version; preserve native fallback bodies."""
    common = replace_once(
        common,
        "  use iso_c_binding, only: c_ptr",
        "  use, intrinsic :: iso_c_binding\n"
        "  use fort_scoped_memory, only: fort_scope_external_binding_v1",
    )
    common = replace_once(
        common,
        "    logical :: in_progress = .false.",
        "\n    logical :: resident_snapshot = .false.",
        after=True,
        insert=True,
    )
    common = replace_once(common, "  interface Add", DECLARATIONS + "\n", after=False, insert=True)
    pos = find_once(common, "        block\n          use, intrinsic :: iso_c_binding")
    _, dosave_hi, _, _ = find_call_span(common, "D%DoSave")
    original = common[pos:dosave_hi]
    replacement = """        block
          use elmm_resident_pressure_state, only: resident_flow
          if (resident_flow%active .and. D%resident_snapshot .and. &
              size(Temperature)==0 .and. size(Moisture)==0 .and. size(Scalar)==0) then
            call QueueResidentFrame(D,U,V,W,Pr,Viscosity)
          else
            call WaitResidentOutput()
""" + original + """
          endif
        end block"""
    common = common[:pos] + replacement + common[dosave_hi:]
    common = replace_once(common, "end module", PROCEDURES + "\n", after=False, insert=True)
    # VTK Fill reads only snapshots plus immutable grid/type geometry in this
    # subset. Exclude LiquidWater/global thermodynamic helper paths explicitly.
    vtk = replace_once(
        vtk,
        "  end subroutine TFrameDomain_SetRest",
        """
    select type(D)
    type is(TFrameDomain)
      D%resident_snapshot = .not.enable_buoyancy .and. .not.enable_moisture .and. &
                            .not.enable_liquid .and. num_of_scalars==0 .and. D%flags%lambda2==0
    class default
      D%resident_snapshot = .false.
    end select
""",
        after=False,
        insert=True,
    )
    lo, hi = find_routine_span(vtk, "SaveVTKFrames")
    body = vtk[lo:hi]
    body = replace_once(
        body,
        "      do i=1,size(FrameDomains)",
        "      call BeginResidentFrameBatch(size(FrameDomains))\n",
        after=False,
        insert=True,
    )
    body = replace_once(
        body,
        "      end do",
        "\n      call FlushResidentFrameBatch()",
        after=True,
        insert=True,
    )
    vtk = vtk[:lo] + body + vtk[hi:]
    main = replace_once(
        main,
        "    use Pressure, only: FinalizeResidentPressure\n    call FinalizeResidentPressure()",
        "    use Frames_common, only: FinalizeResidentOutput\n"
        "    use Pressure, only: FinalizeResidentPressure\n"
        "    call FinalizeResidentOutput()\n    call FinalizeResidentPressure()",
    )
    return common, vtk, main
