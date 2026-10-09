"""Native allocation checks must precede descriptor capture and team decisions."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from elmm_pipeline.instrumentation import region_call


@unittest.skipUnless(shutil.which("gfortran"), "requires GNU Fortran")
class AllocationGuardTests(unittest.TestCase):
    def execute(self, source):
        with tempfile.TemporaryDirectory(prefix="allocation-guards-") as directory:
            root = Path(directory)
            (root / "test.f90").write_text(source)
            built = subprocess.run(
                ["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                 "test.f90", "-o", "test"], cwd=root, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(root / "test")], cwd=root, capture_output=True,
                                 text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS", run.stdout)

    def replacement(self, *, collective=False):
        original = "do i=1,n\na(i)=a(i)+1.0_8\nend do\n"
        if collective:
            original = "!$omp do\n" + original + "!$omp end do\n"
        return region_call({
            "id": "guard_test", "call": "call offload(a,n,lbound(a,1),size(a))",
            "original_text": original,
            "gpu_scope": {"dispatch": "collective_single" if collective else "serial",
                          "expected_omp_level": 1 if collective else 0,
                          "guard": "size(a) >= n", "allocated_arrays": ["a"]},
        }, coverage=False)

    def source(self, body, driver):
        return """module subject
use omp_lib, only: pipeline_level_guard_test=>omp_get_level
implicit none
real(8), allocatable :: a(:)
integer :: calls=0
contains
subroutine offload(field,n,lower,extent)
real(8), intent(inout) :: field(:)
integer, intent(in) :: n,lower,extent
integer :: p
if(size(field)/=extent) error stop 'descriptor mismatch'
calls=calls+1
do p=1,n
field(p-lower+1)=field(p-lower+1)+1.0_8
end do
end subroutine
subroutine advance(n)
integer, intent(in) :: n
integer :: i,lane
""" + body + """
end subroutine
subroutine allocate_late(n)
integer, intent(in) :: n
integer :: first,now,rate
call system_clock(first,rate)
do
call system_clock(now)
if(real(now-first)/real(rate)>0.04) exit
end do
allocate(a(-2:n+2))
a=0.0_8
end subroutine
end module
program check
use subject
use omp_lib, only: omp_set_dynamic
implicit none
call omp_set_dynamic(.false.)
""" + driver + "\nprint *, 'PASS'\nend program\n"

    def test_unallocated_empty_path_and_reallocated_bounds(self):
        self.execute(self.source(self.replacement(), """
call advance(0)
if(calls/=0.or.allocated(a)) error stop 'unallocated empty path'
allocate(a(-2:5)); a=-7.0_8
call advance(3)
if(calls/=1.or.any(a(1:3)/=-6.0_8)) error stop 'allocated path'
if(any(a(-2:0)/=-7.0_8).or.any(a(4:5)/=-7.0_8)) error stop 'halo changed'
deallocate(a)
call advance(0)
if(calls/=1) error stop 'stale allocation state'
allocate(a(1:0))
call advance(0)
if(calls/=2) error stop 'allocated zero-size array'
deallocate(a)
allocate(a(0:9)); a=10.0_8
call advance(7)
if(calls/=3.or.any(a(1:7)/=11.0_8)) error stop 'new allocation bounds'
"""))

    def test_collective_status_observed_after_preceding_nowait(self):
        body = """!$omp parallel num_threads(4) private(i,lane)
!$omp do schedule(static,1)
do lane=1,4
if(lane==4) call allocate_late(n)
end do
!$omp end do nowait
""" + self.replacement(collective=True) + "\n!$omp end parallel\n"
        self.execute(self.source(body, """
call advance(8)
if(calls/=1) error stop 'collective path diverged'
if(any(a(1:8)/=1.0_8)) error stop 'collective data mismatch'
if(any(a(-2:0)/=0.0_8).or.any(a(9:10)/=0.0_8)) error stop 'halo changed'
"""))


if __name__ == "__main__":
    unittest.main()
