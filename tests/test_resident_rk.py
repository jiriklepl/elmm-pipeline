"""Resident boundary forwarding must retain the caller's storage identity."""
import shutil
import subprocess

import pytest

from elmm_pipeline.resident_rk import addressable_arrays, contiguous_periodic_boundary


BOUNDARY = """
module boundary_fixture
  use iso_c_binding
  implicit none
  integer, parameter :: knd = c_double
  type(c_ptr) :: expected(3)
contains
  subroutine BoundUVW_do_BC(U, V, W)
    real(knd), intent(inout), contiguous :: U(-2:,-2:,-2:), V(-2:,-2:,-2:), W(-2:,-2:,-2:)
    call periodic_corners_edges(U, 1)
    call periodic_corners_edges(V, 2)
    call periodic_corners_edges(W, 3)
  end subroutine BoundUVW_do_BC

  subroutine periodic_corners_edges(U, component)
    real(knd), intent(inout) :: U(-2:,-2:,-2:)
    integer, intent(in) :: component
    call bind(U, component)
  end subroutine periodic_corners_edges

  subroutine bind(values, component)
    real(c_double), contiguous, target, intent(in) :: values(:,:,:)
    integer, intent(in) :: component
    if (.not. c_associated(c_loc(values), expected(component))) error stop 'temporary array'
  end subroutine bind
end module boundary_fixture

program test
  use boundary_fixture
  implicit none
  real(c_double), allocatable, target :: U(:,:,:), V(:,:,:), W(:,:,:)
  allocate(U(16,7,7), V(16,7,7), W(16,7,6))
  expected = [c_loc(U), c_loc(V), c_loc(W)]
  call BoundUVW_do_BC(U, V, W)
end program test
"""


@pytest.mark.parametrize("compiler_name", ["gfortran-13", "gfortran-14", "gfortran-15"])
def test_periodic_boundary_preserves_allocation_addresses(tmp_path, compiler_name):
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} unavailable")
    source = tmp_path / "boundary.f90"
    binary = tmp_path / "boundary"
    transformed = addressable_arrays(contiguous_periodic_boundary(BOUNDARY))
    assert "INTENT(INOUT), CONTIGUOUS, TARGET :: U(- 2 :, - 2 :, - 2 :)" in transformed
    source.write_text(transformed.replace(":: values(:,:,:)", ":: values(..)"))
    compiled = subprocess.run([compiler, "-O3", "-frepack-arrays", str(source), "-o", str(binary)],
                              cwd=tmp_path, text=True, capture_output=True, check=False)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("compiler_name", ["gfortran-13", "gfortran-14", "gfortran-15", "gfortran-16"])
def test_periodic_boundary_avoids_packing_at_bind(tmp_path, compiler_name):
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} unavailable")
    source = tmp_path / "boundary.f90"
    for changed in (False, True):
        text = contiguous_periodic_boundary(BOUNDARY) if changed else BOUNDARY
        text = addressable_arrays(text).replace(":: values(:,:,:)", ":: values(..)")
        source.write_text(text)
        command = [compiler, "-O3", "-fopenmp", "-c", "-o", "/dev/null", "-fdump-tree-original=/dev/stdout"]
        command.append(str(source))
        compiled = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=False)
        assert compiled.returncode == 0, compiled.stderr
        periodic = compiled.stdout.split("void periodic_corners_edges", 1)[1].split("void bounduvw_do_bc", 1)[0]
        assert ("_gfortran_internal_pack" in periodic) is not changed


@pytest.mark.parametrize("source", [
    BOUNDARY.replace("intent(inout), contiguous :: U", "intent(inout) :: U"),
    BOUNDARY.replace("call periodic_corners_edges(W, 3)", "call periodic_corners_edges(W, 2)"),
    BOUNDARY.replace("    call periodic_corners_edges(W, 3)\n", "").replace(
        "  call BoundUVW_do_BC(U, V, W)",
        "  call periodic_corners_edges(W, 3)\n  call BoundUVW_do_BC(U, V, W)",
    ),
])
def test_periodic_boundary_rejects_unproven_callers(source):
    with pytest.raises(ValueError):
        contiguous_periodic_boundary(source)
