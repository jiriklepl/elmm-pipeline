"""Resident filter forwarding must retain the caller's storage identity."""
import shutil
import subprocess

import pytest

from compiler.scopes.resident_splice import addressable_arrays
from elmm_pipeline.resident_rk import contiguous_filter_arguments


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


def test_configured_sources_skips_include_fragments_and_discover_excludes_inc(tmp_path):
    from compiler.resident.configure import configure_sources
    from elmm_pipeline.resident_build import discover_application_units

    src = tmp_path / "src"
    src.mkdir()
    (src / "list-inc-def.f90").write_text("  type list_node\n    TYPEPARAM :: item\n  end type\n")
    (src / "list-inc-proc.f90").write_text("  subroutine add(item)\n    TYPEPARAM :: item\n  end subroutine\n")
    (src / "mod.f90").write_text("module m\ncontains\nsubroutine s()\nend subroutine\nend module\n")
    units, _ = discover_application_units(src)
    assert units == ["mod"]
    gfortran = shutil.which("gfortran") or shutil.which("gfortran-15")
    if gfortran:
        configured = configure_sources([src / "list-inc-def.f90", src / "mod.f90"], ["m::s"],
                                       tmp_path / "configured", fortran=gfortran)
        assert len(configured["sources"]) == 2
