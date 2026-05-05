"""Tests for gromacs-mcp tools.

Tests that don't call GROMACS directly run always.
Tests marked requires_gromacs are skipped when gmx is not in PATH.
"""
import json
from pathlib import Path

import pytest

from gromacs_mcp import templates
from gromacs_mcp.gmx import find_gmx


# ---------------------------------------------------------------------------
# Template rendering (no GROMACS needed)
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_em_renders(self):
        mdp = templates.render_em()
        assert "steep" in mdp
        assert "50000" in mdp

    def test_em_custom_steps(self):
        mdp = templates.render_em(max_steps=10000)
        assert "10000" in mdp

    def test_nvt_nsteps(self):
        mdp = templates.render_nvt(duration_ps=200.0)
        # 200 ps / 0.002 ps = 100000 steps
        assert "100000" in mdp

    def test_nvt_temperature(self):
        mdp = templates.render_nvt(temperature=350.0)
        assert "350.0" in mdp

    def test_nvt_single_group(self):
        mdp = templates.render_nvt(tc_groups=["System"])
        assert "System" in mdp

    def test_nvt_two_groups(self):
        mdp = templates.render_nvt(tc_groups=["BICH", "SOL"])
        assert "BICH SOL" in mdp
        assert "0.1   0.1" in mdp

    def test_npt_has_barostat(self):
        mdp = templates.render_npt()
        assert "C-rescale" in mdp
        assert "semiisotropic" in mdp

    def test_production_has_nose_hoover(self):
        mdp = templates.render_production()
        assert "Nose-Hoover" in mdp
        assert "Parrinello-Rahman" in mdp

    def test_production_default_duration(self):
        mdp = templates.render_production()
        # 300 ps / 0.002 = 150000 steps
        assert "150000" in mdp

    def test_production_custom_duration(self):
        mdp = templates.render_production(duration_ps=500.0)
        # 500 / 0.002 = 250000
        assert "250000" in mdp


# ---------------------------------------------------------------------------
# GROMACS availability
# ---------------------------------------------------------------------------


@pytest.mark.requires_gromacs
class TestGromacsAvailable:
    def test_find_gmx(self):
        path = find_gmx()
        assert path is not None
        assert "gmx" in path

    def test_check_gromacs_tool(self):
        from gromacs_mcp.server import check_gromacs

        result = check_gromacs()
        assert result["available"] is True
        assert result["version"]


# ---------------------------------------------------------------------------
# End-to-end workflow (requires biochar .gro/.top + GROMACS)
# ---------------------------------------------------------------------------


@pytest.mark.requires_gromacs
class TestWorkflow:
    """Integration test using the example biochar structure from the biochar package."""

    @pytest.fixture
    def biochar_files(self, tmp_path):
        """Copy test fixtures into a temp directory."""
        fixtures = Path(__file__).parent / "fixtures"
        gro = fixtures / "biochar.gro"
        top = fixtures / "biochar.top"
        if not gro.exists() or not top.exists():
            pytest.skip("No test fixtures found at tests/fixtures/")
        import shutil

        work = tmp_path / "work"
        work.mkdir()
        shutil.copy(gro, work / "biochar.gro")
        shutil.copy(top, work / "biochar.top")
        return work / "biochar.gro", work / "biochar.top", work

    def test_energy_minimize(self, biochar_files):
        from gromacs_mcp.server import energy_minimize

        gro, top, outdir = biochar_files
        result = energy_minimize(
            gro_path=str(gro),
            top_path=str(top),
            output_dir=str(outdir / "em"),
            max_steps=500,
        )
        assert result["success"], result.get("error")
        assert Path(result["minimized_gro"]).exists()
