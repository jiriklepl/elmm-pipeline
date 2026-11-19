from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from compiler.scopes.resident_entries import export_resident_entries
from elmm_pipeline.ast_rewrites import (
    expand_pure_calls_in_text,
    find_all_calls,
    find_call_span,
    find_routine_span,
    inline_pure_scalar_function,
    replace_once,
)
from elmm_pipeline.gpu_hot_path import (
    _divergence_expression,
    materialize_pressure_stencils,
)
from elmm_pipeline.resident_build import (
    ELMM_UNITS,
    POISFFT_UNITS,
    RK_PROCEDURES,
    deterministic_reference,
    diagnostic_main,
    discover_application_units,
    discover_resident_procedure_groups,
)
from elmm_pipeline.resident_rk import (
    contiguous_filter_arguments,
    contiguous_periodic_boundary,
    pointwise_advection,
    snapshot_filter,
)
from elmm_pipeline.resident_walls import _verify_wall_helpers


ELMM_SRC = (
    Path(__file__).resolve().parents[1] / ".cache" / "repos" / "elmm" / "src"
)
POISFFT_SRC = (
    Path(__file__).resolve().parents[1] / ".cache" / "repos" / "PoisFFT" / "src"
)


def test_deterministic_reference_handles_formatting_variants() -> None:
    variant = """module press_mod
  ! modified indentation and extra comments
  implicit none
contains
  subroutine FixPressureToReference(Pr)
    !$omp   do   reduction( + : Phi_ref )   ! ref plane sum
    do   j = 1, Prny
      do  i = 1, Prnx
        Phi_ref = Phi_ref + Pr( i , j , k_ref )
      enddo
    enddo
    !$omp end do

    ! optional comment between directives
    !$omp   single
    Phi_ref = Phi_ref / ( Prnx * Prny )
    !$omp   end   single
  end subroutine FixPressureToReference
end module press_mod
"""
    rewritten = deterministic_reference(variant)
    assert "!$omp master" in rewritten
    assert "!$omp barrier" in rewritten
    assert "reduction" not in rewritten
    # Idempotent once already rewritten.
    assert deterministic_reference(rewritten) == rewritten

    with pytest.raises(ValueError):
        deterministic_reference("subroutine empty()\nend subroutine empty\n")


def test_pointwise_advection_handles_renamed_routines_and_arrays(tmp_path: Path) -> None:
    source = tmp_path / "custom_adv.f90"
    source.write_text(
        """module custom_adv_mod
  implicit none
contains
  subroutine custom_div_x(nx, ny, nz, ts, dx, flux_a, out_arr)
    integer, intent(in) :: nx, ny, nz, ts
    real(8), intent(in) :: dx
    real(8), intent(in) :: flux_a(0:nx+1, 0:ny+1, 0:nz+1)
    real(8), intent(out) :: out_arr(1:nx, 1:ny, 1:nz)
    real(8) :: work_u(0:ts+1, 0:ts+1, 0:ts+1)
    integer :: i, j, k, it, jt, kt
    integer :: lx, ly, lz, cx, cy, cz

    do kt = 1, nz, ts
      do jt = 1, ny, ts
        do it = 1, nx, ts
          do k = kt, min(kt + ts - 1, nz)
            lz = k - kt + 1
            do j = jt, min(jt + ts - 1, ny)
              ly = j - jt + 1
              do i = it - 1, min(it + ts - 1, nx)
                lx = i - it + 1
                work_u(lx, ly, lz) = 0.5d0 * (flux_a(i, j, k) + flux_a(i + 1, j, k))
              end do
            end do
          end do
          do k = kt, min(kt + ts - 1, nz)
            cz = k - kt + 1
            do j = jt, min(jt + ts - 1, ny)
              cy = j - jt + 1
              do i = it, min(it + ts - 1, nx)
                cx = i - it + 1
                out_arr(i, j, k) = (work_u(cx, cy, cz) - work_u(cx - 1, cy, cz)) / dx
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine custom_div_x
end module custom_adv_mod
""",
        encoding="utf-8",
    )

    result = pointwise_advection(source, tmp_path / "out")
    transformed = Path(result["replacement"]["path"]).read_text(encoding="utf-8")
    assert "work_u(" not in transformed.split("integer :: lx, ly, lz, cx, cy, cz", 1)[-1]
    assert "0.5d0" in transformed.lower()
    assert result["directional_updates"] == 1


def test_inline_tile_scratch_nests_on_elmm() -> None:
    from compiler.resident.rewrites import inline_tile_scratch_nests

    if not (ELMM_SRC / "advection.f90").is_file():
        pytest.skip("ELMM source checkout unavailable")
    adv_text = (ELMM_SRC / "advection.f90").read_text(encoding="utf-8")
    # Inject extra comments to confirm formatting/comment invariance.
    perturbed = adv_text.replace("end do", "! loop end\nend do", 5)
    _, changes = inline_tile_scratch_nests(perturbed)
    assert len(changes) == 9


def test_snapshot_filter_resilient_to_formatting_and_comments(tmp_path: Path) -> None:
    source = tmp_path / "filters_mod.f90"
    source.write_text(
        """module filters_mod
  implicit none
contains
  subroutine Filter3ord(U, Utype, dir)
    real(8), dimension(-2:, -2:, -2:), intent(inout) :: U
    integer, dimension(-2:, -2:, -2:), intent(in) :: Utype
    integer, intent(in) :: dir
    real(8), parameter :: w(-3:3) = [ -1._8/32, 0._8, 9._8/32, 0.5_8, 9._8/32, 0._8, -1._8/32 ]
    real(8) :: S, q
    integer :: i, j, k, ii, jj, kk
    integer :: mini, minj, mink, maxi, maxj, maxk
    real(8) :: tmp(-2:ubound(U,dir))

    mini = lbound(U, 1) + 3
    maxi = ubound(U, 1) - 3
    minj = lbound(U, 2) + 3
    maxj = ubound(U, 2) - 3
    mink = lbound(U, 3) + 3
    maxk = ubound(U, 3) - 3

    if (dir == 1) then
      ! formatting variant comment 1
      do k = mink, maxk
        do j = minj, maxj
          tmp = U(:, j, k)
          do i = 1, maxi
            S = 0._8
            q = 0._8
            do ii = i - 3, i + 3
              if (Utype(ii, j, k) <= 0) then
                S = S + w(ii - i) * tmp(ii)
                q = q + w(ii - i)
              end if
            end do
            if (abs(q) > 0.74_8) U(i, j, k) = S / q
          enddo
        enddo
      enddo
    else if (dir == 2) then
      ! formatting variant comment 2
      do k = mink, maxk
        do i = mini, maxi
          tmp = U(i, :, k)
          do j = 1, maxj
            S = 0._8
            q = 0._8
            do jj = j - 3, j + 3
              if (Utype(i, jj, k) <= 0) then
                S = S + w(jj - j) * tmp(jj)
                q = q + w(jj - j)
              end if
            end do
            if (abs(q) > 0.74_8) U(i, j, k) = S / q
          enddo
        enddo
      enddo
    else
      do j = minj, maxj
        do i = mini, maxi
          tmp = U(i, j, :)
          do k = 1, maxk
            S = 0._8
            q = 0._8
            do kk = k - 3, k + 3
              if (Utype(i, j, kk) <= 0) then
                S = S + w(kk - k) * tmp(kk)
                q = q + w(kk - k)
              end if
            end do
            if (abs(q) > 0.74_8) U(i, j, k) = S / q
          enddo
        enddo
      enddo
    end if
  end subroutine Filter3ord
end module filters_mod
""",
        encoding="utf-8",
    )
    result = snapshot_filter(source, tmp_path / "out")
    text = Path(result["replacement"]["path"]).read_text(encoding="utf-8")
    assert "allocatable :: tmp(:,:,:)" in text
    assert "allocate(tmp(" in text
    assert "tmp((i+ii),j,k)" in text
    assert "tmp(i,(j+jj),k)" in text
    assert "tmp(i,j,(k+kk))" in text
    assert "resident_flow%release_real(tmp)" in text


def test_divergence_expression_and_pressure_stencils_renamed_args(tmp_path: Path) -> None:
    operators = tmp_path / "numerical_operators.f90"
    operators.write_text(
        """module operators_mod
  implicit none
contains
  pure real(8) function Divergence4Uniform(um2, um1, up1, up2, &
      vm2, vm1, vp1, vp2, &
      wm2, wm1, wp1, wp2, &
      dx, dy, dz) result(div_val)
    real(8), intent(in) :: um2, um1, up1, up2
    real(8), intent(in) :: vm2, vm1, vp1, vp2
    real(8), intent(in) :: wm2, wm1, wp1, wp2
    real(8), intent(in) :: dx, dy, dz
    real(8), parameter :: c1 = 27.0d0 / 24.0d0
    real(8), parameter :: c3 = 1.0d0 / 24.0d0
    div_val = (c1 * (up1 - um1) - c3 * (up2 - um2)) / dx + &
              (c1 * (vp1 - vm1) - c3 * (vp2 - vm2)) / dy + &
              (c1 * (wp1 - wm1) - c3 * (wp2 - wm2)) / dz
  end function Divergence4Uniform
end module operators_mod
""",
        encoding="utf-8",
    )
    expr, constants = _divergence_expression(operators)
    assert "c1" in constants and "c3" in constants
    assert "dxmin" in expr and "dymin" in expr and "dzmin" in expr

    renamed_ops = tmp_path / "renamed_ops.f90"
    renamed_ops.write_text(
        operators.read_text(encoding="utf-8")
        .replace("dx, dy, dz", "hx, hy, hz")
        .replace(":: dx, dy, dz", ":: hx, hy, hz")
        .replace("c1", "w1")
        .replace("c3", "w3")
        .replace("/ dx", "/ hx")
        .replace("/ dy", "/ hy")
        .replace("/ dz", "/ hz"),
        encoding="utf-8",
    )
    parsed = inline_pure_scalar_function(
        renamed_ops.read_text(encoding="utf-8"), "Divergence4Uniform"
    )
    call_text = """
    Pr(i, j, k) = Divergence4Uniform( &
      U(i-1,j,k), U(i,j,k), U(i+1,j,k), U(i+2,j,k), &
      V(i,j-1,k), V(i,j,k), V(i,j+1,k), V(i,j+2,k), &
      W(i,j,k-1), W(i,j,k), W(i,j,k+1), W(i,j,k+2), &
      dxmin, dymin, dzmin)
    """
    expanded_text, expanded_count = expand_pure_calls_in_text(call_text, parsed)
    assert expanded_count == 1
    assert "hx" not in expanded_text
    assert "dxmin" in expanded_text and "U(i+1,j,k)" in expanded_text

    if (ELMM_SRC / "pressure.f90").is_file() and (ELMM_SRC / "numerical_operators.f90").is_file():
        work_src = tmp_path / "elmm_copy"
        work_src.mkdir()
        shutil.copyfile(ELMM_SRC / "pressure.f90", work_src / "pressure.f90")
        ops_text = (ELMM_SRC / "numerical_operators.f90").read_text(encoding="utf-8")
        # Rename Divergence4Uniform dummy spacing args if matched
        ops_mod = ops_text.replace("dx,dy,dz", "hx,hy,hz").replace("dx, dy, dz", "hx, hy, hz")
        (work_src / "numerical_operators.f90").write_text(ops_mod, encoding="utf-8")
        stencils = materialize_pressure_stencils(
            work_src / "pressure.f90", tmp_path / "stencils_out"
        )
        assert Path(stencils["replacement"]["path"]).is_file()


def test_verify_wall_helpers_resilient_to_comments_and_rejects_missing() -> None:
    if not (ELMM_SRC / "wallmodels.f90").is_file():
        pytest.skip("ELMM snapshot missing")
    wall_src = (ELMM_SRC / "wallmodels.f90").read_text(encoding="utf-8")
    modified = wall_src.replace(
        "subroutine u_log_law_2(",
        "! formatting change in wall model helper\nsubroutine u_log_law_2(",
        1,
    ).replace(
        "end subroutine",
        "  ! trailing comment\nend subroutine",
    )
    digests = _verify_wall_helpers(modified)
    assert len(digests) == 6
    assert all(len(v) == 64 for v in digests.values())

    corrupted = wall_src.replace("WMFlatUstar", "WMFlatUstar_missing")
    with pytest.raises(ValueError):
        _verify_wall_helpers(corrupted)


def test_contiguous_checks_handle_spaced_multiline_declarations() -> None:
    good_filter = """module m_filt
  implicit none
contains
  subroutine Filter3ord(U, Utype, dir)
    real(kind=8), dimension(-2:, -2:, -2:), intent(inout) :: U
    integer, dimension(-2:, -2:, -2:), intent(in) :: Utype
    integer, intent(in) :: dir
    print *, dir, size(U), size(Utype)
  end subroutine Filter3ord
  subroutine Filter(U, Utype)
    real(8), intent(inout) :: U(-2:, -2:, -2:)
    integer, intent(in) :: Utype(-2:, -2:, -2:)
    call Filter3ord(U, Utype, 1)
  end subroutine Filter
end module m_filt
"""
    rewritten_filt = contiguous_filter_arguments(good_filter)
    assert rewritten_filt.count("contiguous") == 4

    with pytest.raises(ValueError):
        contiguous_filter_arguments(good_filter.replace(", intent(in) :: Utype(-2:, -2:, -2:)", ", intent(in) :: Utype(:,:,:)"))

    good_bound = """module m_bnd
  implicit none
contains
  subroutine BoundUVW_do_BC(U, V, W)
    real(8), contiguous, intent(inout) :: U(-2:, -2:, -2:), &
      V(-2:, -2:, -2:), W(-2:, -2:, -2:)
    call periodic_corners_edges(U, 1)
    call periodic_corners_edges(V, 2)
    call periodic_corners_edges(W, 3)
  end subroutine BoundUVW_do_BC
  subroutine periodic_corners_edges(U, dir)
    real(8), dimension(-2:, -2:, -2:), intent(inout) :: U
    integer, intent(in) :: dir
    print *, dir, size(U)
  end subroutine periodic_corners_edges
end module m_bnd
"""
    rewritten_bnd = contiguous_periodic_boundary(good_bound)
    assert "contiguous" in rewritten_bnd.split("subroutine periodic_corners_edges", 1)[1]

    bad_bound = good_bound + """
subroutine extra_caller(X)
  real(8), intent(inout) :: X(-2:,-2:,-2:)
  call periodic_corners_edges(X, 1)
end subroutine extra_caller
"""
    with pytest.raises(ValueError):
        contiguous_periodic_boundary(bad_bound)


def test_ast_call_lookup_and_diagnostic_main() -> None:
    code = """module main_test
  implicit none
contains
  subroutine run_step(a, b, c)
    integer :: a, b, c
    call compute_flux( &
      a, &
      b, c)
    if (.true.) call finalize_outputs()
  end subroutine run_step
end module main_test
"""
    span = find_call_span(code, "compute_flux", within_routine="run_step")
    assert "compute_flux" in span[2].lower()
    assert span[3] == ["a", "b", "c"]
    assert len(find_all_calls(code, "finalize_outputs")) == 1

    rstart, rend = find_routine_span(code, "run_step")
    assert code[rstart:rend].lstrip().lower().startswith("subroutine run_step")

    diag = diagnostic_main(
        """program main
  print *, "start"
  call Dynamics_Deallocate
  call DeallocateGlobals
end program main
""",
        resident=False,
    )
    assert "fields-U.bin" in diag
    assert "fields-U_avg.bin" in diag
    assert "replace_once" in replace_once.__name__


def test_discovery_and_wildcard_export(tmp_path: Path) -> None:
    if ELMM_SRC.is_dir():
        elmm_units, poisfft_units = discover_application_units(
            ELMM_SRC, POISFFT_SRC if POISFFT_SRC.is_dir() else None
        )
        assert set(elmm_units) == set(ELMM_UNITS)
        assert len(elmm_units) == len(ELMM_UNITS)
        if POISFFT_SRC.is_dir():
            assert set(poisfft_units) == set(POISFFT_UNITS)
            assert len(poisfft_units) == len(POISFFT_UNITS)

        groups = discover_resident_procedure_groups(
            [ELMM_SRC / f"{unit}.f90" for unit in elmm_units]
        )
        assert set(groups) == {"rk", "boundary", "periodic", "advection", "filter"}
        assert set(groups["rk"]) == set(RK_PROCEDURES)
        assert "filters::filter3ord" in groups["filter"]
        assert "boundaries::bounduvw_do_bc" in groups["boundary"]
        assert "boundaries::periodic_corners_edges" in groups["periodic"]

    # Synthetic SConstruct-less directory discovery + wildcard export_resident_entries
    synth = tmp_path / "synth_src"
    synth.mkdir()
    mod_file = synth / "alpha.f90"
    mod_file.write_text(
        """module alpha_mod
  implicit none
contains
  subroutine step_one(n, a)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(1:n)
    integer :: i
    do i = 1, n
      a(i) = a(i) + 1.0d0
    end do
  end subroutine step_one
  subroutine step_two(n, b)
    integer, intent(in) :: n
    real(8), intent(inout) :: b(1:n)
    integer :: i
    do i = 1, n
      b(i) = b(i) * 2.0d0
    end do
  end subroutine step_two
end module alpha_mod
""",
        encoding="utf-8",
    )
    units, pois = discover_application_units(synth)
    assert units == ["alpha"]
    assert pois == []

    exported = export_resident_entries(
        [mod_file],
        ["alpha_mod::*"],
        tmp_path / "entries",
        allow_patterns=True,
    )
    assert [e["procedure"] for e in exported["entries"]] == ["alpha_mod::step_one", "alpha_mod::step_two"]
    assert len(exported["entries"]) == 2

    exported_all = export_resident_entries(
        [mod_file],
        None,
        tmp_path / "entries_all",
    )
    assert [e["procedure"] for e in exported_all["entries"]] == ["alpha_mod::step_one", "alpha_mod::step_two"]


def test_pressure_tail_and_workspace_resilience(tmp_path: Path) -> None:
    from elmm_pipeline.gpu_hot_path import _verify_pressure_tail_routines
    from elmm_pipeline.resident_workspace import integrate_workspace

    if (ELMM_SRC / "pressure.f90").is_file() and (ELMM_SRC / "boundaries.f90").is_file():
        p_text = deterministic_reference((ELMM_SRC / "pressure.f90").read_text(encoding="utf-8"))
        b_text = (ELMM_SRC / "boundaries.f90").read_text(encoding="utf-8")
        # Inject comments and spacing changes into FixPressureToReference and Bound_Pr
        p_mod = p_text.replace(
            "subroutine FixPressureToReference",
            "! comment before routine\n  subroutine   FixPressureToReference",
            1,
        )
        b_mod = b_text.replace(
            "subroutine Bound_Pr",
            "! comment before Bound_Pr\n  subroutine   Bound_Pr",
            1,
        )
        ref_sha, face_sha = _verify_pressure_tail_routines(p_mod, b_mod)
        assert len(ref_sha) == 64 and len(face_sha) == 64

        with pytest.raises(ValueError):
            _verify_pressure_tail_routines("module m\nend module m\n", b_mod)

    # Verify integrate_workspace handles non-canonical real(8) allocatable declaration in Filter3ord
    tsteps = tmp_path / "tsteps.f90"
    tsteps.write_text(
        """module TimeSteps
  implicit none
contains
  subroutine TMarchRK3()
    integer :: RK_stage
    call step_rk( &
                      RK_beta, RK_rho, RK_stage, time_stepping%dt)
  end subroutine TMarchRK3
end module TimeSteps
""",
        encoding="utf-8",
    )
    dynamics = tmp_path / "dynamics.f90"
    dynamics.write_text(
        """module Dynamics
  implicit none
contains
  subroutine Convection()
    block
      use elmm_resident_pressure_state, only: resident_flow
      call resident_flow%release_real(beta)
      call resident_flow%release_real(rho)
    end block
  end subroutine Convection
end module Dynamics
""",
        encoding="utf-8",
    )
    filters = tmp_path / "filters.f90"
    filters.write_text(
        """module Filters
  implicit none
contains
  subroutine Filter3ord(U, Utype, dir)
    real(8), allocatable, target :: tmp(:,:,:)
    allocate(tmp(-2:ubound(U,1),-2:ubound(U,2),-2:ubound(U,3)))
    rk_handle_tmp = resident_flow%bind_real64(tmp, lbound(tmp), .true., &
      origin='filters::filter3ord::tmp')
    rk_handle_tmp = resident_flow%bind_real64(tmp, lbound(tmp), .true., &
      origin='filters::filter3ord::tmp')
    rk_handle_tmp = resident_flow%bind_real64(tmp, lbound(tmp), .true., &
      origin='filters::filter3ord::tmp')
    rk_handle_tmp = resident_flow%bind_real64(tmp, lbound(tmp), .true., &
      origin='filters::filter3ord::tmp')
    block
      use elmm_resident_pressure_state, only: resident_flow
      call resident_flow%release_real(tmp)
    end block
  end subroutine Filter3ord
end module Filters
""",
        encoding="utf-8",
    )
    ws = integrate_workspace(
        {"sources": [{"path": str(tsteps)}, {"path": str(dynamics)}, {"path": str(filters)}]},
        tmp_path / "ws_out",
    )
    filt_out = (tmp_path / "ws_out" / "filters.f90").read_text(encoding="utf-8")
    assert "acquire_work_real_3" in filt_out
    assert "release_work_real_3" in filt_out
    assert len(ws["sources"]) == 3
