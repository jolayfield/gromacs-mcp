"""Tests for gromacs-mcp tools.

Tests that don't call GROMACS directly run always.
Tests marked requires_gromacs are skipped when gmx is not in PATH.
"""
import shutil
from pathlib import Path

import pytest

from gromacs_mcp import templates
from gromacs_mcp.gmx import count_ndx_groups, find_gmx

FIXTURES = Path(__file__).parent / "fixtures"


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
        # 200 ps / 0.002 ps = 100000 steps
        assert "100000" in templates.render_nvt(duration_ps=200.0)

    def test_nvt_temperature(self):
        assert "350.0" in templates.render_nvt(temperature=350.0)

    def test_nvt_single_group(self):
        assert "System" in templates.render_nvt(tc_groups=["System"])

    def test_nvt_two_groups(self):
        mdp = templates.render_nvt(tc_groups=["Biochar", "Water"])
        assert "Biochar Water" in mdp
        assert "0.1 0.1" in mdp

    def test_npt_has_barostat(self):
        mdp = templates.render_npt()
        assert "C-rescale" in mdp
        assert "semiisotropic" in mdp

    def test_production_has_nose_hoover(self):
        mdp = templates.render_production()
        assert "Nose-Hoover" in mdp
        assert "Parrinello-Rahman" in mdp

    def test_production_default_duration(self):
        # 300 ps / 0.002 = 150000 steps
        assert "150000" in templates.render_production()

    def test_production_custom_duration(self):
        # 500 / 0.002 = 250000
        assert "250000" in templates.render_production(duration_ps=500.0)


class TestNdxGroupParsing:
    def test_parses_typical_output(self):
        output = """
  0 System              :   24 atoms
  1 BC                  :   16 atoms
  2 non-Water           :   24 atoms
"""
        assert count_ndx_groups(output) == 3

    def test_empty_output_returns_zero(self):
        assert count_ndx_groups("no groups here") == 0

    def test_single_group(self):
        assert count_ndx_groups("  0 System : 10 atoms") == 1


# ---------------------------------------------------------------------------
# GROMACS availability
# ---------------------------------------------------------------------------


@pytest.mark.requires_gromacs
class TestGromacsAvailable:
    def test_find_gmx(self):
        path = find_gmx()
        assert "gmx" in path

    def test_check_gromacs_tool(self):
        from gromacs_mcp.server import check_gromacs

        result = check_gromacs()
        assert result["available"] is True
        assert result["version"]


# ---------------------------------------------------------------------------
# Integration tests (require GROMACS + fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.requires_gromacs
class TestWorkflow:
    """Integration tests using the committed biochar fixture."""

    @pytest.fixture
    def work(self, tmp_path):
        gro = FIXTURES / "biochar.gro"
        top = FIXTURES / "biochar.top"
        if not gro.exists():
            pytest.skip("Fixture tests/fixtures/biochar.gro not found")
        d = tmp_path / "work"
        d.mkdir()
        for name in ("biochar.gro", "biochar.top", "biochar.itp"):
            src = FIXTURES / name
            if src.exists():
                shutil.copy(src, d / name)
        return d

    def test_energy_minimize(self, work):
        from gromacs_mcp.server import energy_minimize
        from gromacs_mcp.gmx import run

        # Set a cubic box with 1.5 nm padding — required before EM with PME cutoffs
        run("editconf", "-f", str(work / "biochar.gro"),
            "-o", str(work / "biochar.gro"), "-box", "4", "4", "4", "-c")

        result = energy_minimize(
            gro_path=str(work / "biochar.gro"),
            top_path=str(work / "biochar.top"),
            output_dir=str(work / "em"),
            max_steps=500,
        )
        assert result["success"], result.get("error")
        assert Path(result["minimized_gro"]).exists()

    def test_make_index_default_groups(self, work):
        from gromacs_mcp.server import make_index

        result = make_index(
            gro_path=str(work / "biochar.gro"),
            output_dir=str(work / "ndx"),
            groups=[],
        )
        assert result["success"], result.get("error")
        assert Path(result["ndx_path"]).exists()

    def test_make_index_custom_group(self, work):
        from gromacs_mcp.server import make_index

        result = make_index(
            gro_path=str(work / "biochar.gro"),
            output_dir=str(work / "ndx"),
            groups=[{"residue": "BC", "name": "Biochar"}],
        )
        assert result["success"], result.get("error")
        assert "Biochar" in result["custom_groups"]
        ndx_text = Path(result["ndx_path"]).read_text()
        assert "Biochar" in ndx_text
