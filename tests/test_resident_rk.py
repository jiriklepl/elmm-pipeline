"""Resident boundary forwarding must retain the caller's storage identity."""
import shutil
import subprocess

import pytest

from elmm_pipeline.resident_rk import (
    addressable_arrays, contiguous_filter_arguments, contiguous_periodic_boundary, identity, materialize_entries,
)


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


FILTER = """
module filter_fixture
  use iso_c_binding
  implicit none
  integer, parameter :: knd = c_double
  type(c_ptr) :: real_address, integer_address
contains
  subroutine Filter3ord(U, Utype, dir)
    real(knd), dimension(-2:,-2:,-2:), intent(inout)  :: U
    integer, dimension(-2:,-2:,-2:), intent(in)  :: Utype
    integer, intent(in) :: dir
    call bind_real(U)
    call bind_integer(Utype)
  end subroutine Filter3ord
  subroutine Filter(U, Utype)
    real(knd), dimension(-2:,-2:,-2:), intent(inout) :: U
    integer, dimension(-2:,-2:,-2:), intent(in)  :: Utype
    call Filter3ord(U, Utype, 1)
  end subroutine Filter
  subroutine bind_real(values)
    real(c_double), contiguous, target, intent(in) :: values(:,:,:)
    if (.not. c_associated(c_loc(values), real_address)) error stop 'real temporary'
  end subroutine bind_real
  subroutine bind_integer(values)
    integer(c_int), contiguous, target, intent(in) :: values(:,:,:)
    if (.not. c_associated(c_loc(values), integer_address)) error stop 'integer temporary'
  end subroutine bind_integer
end module filter_fixture
program test
  use filter_fixture
  real(c_double), allocatable, target :: U(:,:,:)
  integer(c_int), allocatable, target :: Utype(:,:,:)
  allocate(U(16,7,7), Utype(16,7,7))
  real_address=c_loc(U)
  integer_address=c_loc(Utype)
  call Filter(U, Utype)
end program test
"""


@pytest.mark.parametrize("compiler_name", ["gfortran-13", "gfortran-14", "gfortran-15", "gfortran-16"])
def test_filter_bindings_preserve_addresses_without_packing(tmp_path, compiler_name):
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} unavailable")
    source = tmp_path / "filter.f90"
    binary = tmp_path / "filter"
    for changed in (False, True):
        text = contiguous_filter_arguments(FILTER) if changed else FILTER
        text = addressable_arrays(text).replace(":: values(:,:,:)", ":: values(..)")
        source.write_text(text)
        command = [compiler, "-O3", "-fopenmp", "-fdump-tree-original=/dev/stdout", str(source), "-o", str(binary)]
        compiled = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=False)
        assert compiled.returncode == 0, compiled.stderr
        filter3 = compiled.stdout.split("void filter3ord", 1)[1].split("void filter ", 1)[0]
        assert ("_gfortran_internal_pack" in filter3) is not changed
        if changed:
            result = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
            assert result.returncode == 0, result.stderr


def test_filter_rejects_unexpected_array_contract():
    with pytest.raises(ValueError, match="Unsupported filter array declaration"):
        contiguous_filter_arguments(FILTER.replace("intent(in)  :: Utype", "intent(inout)  :: Utype"))


def test_generated_bindings_include_source_origins(tmp_path):
    source = tmp_path / "example.f90"
    source.write_text("subroutine update(u, mask)\n  u=1\nend subroutine\n")
    region = {
        "source": str(source), "first_line": 2, "last_line": 2, "procedure": "example::update",
        "resources": [
            {"resource": "argument::u", "name": "u", "type": "real", "kind": 8},
            {"resource": "argument::mask", "name": "mask", "type": "integer", "kind": 4},
        ],
        "parameters": [
            {"resource": "argument::u", "name": "u", "rank": 3, "lower_bound_dimension": None},
            {"resource": "argument::mask", "name": "mask", "rank": 3, "lower_bound_dimension": None},
        ],
        "runtime_guards": [],
    }
    entry = {"fortran_module": "fort_example", "argument_order": ["context", "device", "u", "mask"]}
    package = {"source_inputs": {str(source): identity(source)["sha256"]},
               "entries": [{"region": region, "entry": entry}], "rejected": []}
    result = materialize_entries([package], tmp_path / "generated")
    generated = (tmp_path / "generated" / source.name).read_text()
    assert len(result["sources"]) == 1
    assert "bind_real64(u,lbound(u),.true., &\n          origin='example::update::u')" in generated
    assert "bind_integer32(mask,lbound(mask),.true., &\n          origin='example::update::mask')" in generated
