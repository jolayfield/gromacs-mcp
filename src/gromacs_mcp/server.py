"""GROMACS MCP server.

Exposes each stage of a standard MD pipeline as Claude-callable tools:
  check_gromacs → solvate → energy_minimize → equilibrate_nvt
  → equilibrate_npt → run_production → analyze_*

All path arguments accept absolute or CWD-relative paths.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import analysis, gmx, templates

mcp = FastMCP(
    "gromacs-mcp",
    instructions=(
        "Run GROMACS molecular dynamics simulations on molecular structures. "
        "Standard workflow: solvate → energy_minimize → equilibrate_nvt → "
        "equilibrate_npt → run_production → analyze_*. "
        "Each tool returns output file paths for use in the next step."
    ),
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@mcp.tool()
def check_gromacs() -> dict:
    """Check whether GROMACS is available and return its version string."""
    result = gmx.run("--version")
    version = next(
        (l.strip() for l in result["stdout"].splitlines() if "GROMACS version" in l),
        result["stdout"][:200],
    )
    return {"available": result["success"], "version": version}


# ---------------------------------------------------------------------------
# System preparation
# ---------------------------------------------------------------------------


@mcp.tool()
def solvate(
    gro_path: str,
    top_path: str,
    output_dir: str,
    water_model: str = "spc216",
    box_padding_nm: float = 1.0,
) -> dict:
    """
    Add a periodic water box around a dry structure.

    Calls gmx editconf (to set box size) then gmx solvate.
    The topology file is updated in-place with the water molecule count.

    Args:
        gro_path: Dry input structure (.gro).
        top_path: Topology file (.top) — updated in-place.
        output_dir: Directory to write solvated.gro.
        water_model: Water configuration file ("spc216" or "tip4p").
        box_padding_nm: Minimum distance from solute to box edge (nm).

    Returns:
        Paths to solvated structure and topology, plus water count added.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    boxed_gro = outdir / "boxed.gro"
    solvated_gro = outdir / "solvated.gro"

    box_result = gmx.run(
        "editconf",
        "-f", gro_path,
        "-o", str(boxed_gro),
        "-d", str(box_padding_nm),
        "-bt", "cubic",
    )
    if not box_result["success"]:
        return {"success": False, "error": box_result["stderr"], "step": "editconf"}

    sol_result = gmx.run(
        "solvate",
        "-cp", str(boxed_gro),
        "-cs", water_model,
        "-o", str(solvated_gro),
        "-p", top_path,
    )

    n_water = 0
    for line in sol_result["stdout"].splitlines():
        if "Number of solvent molecules" in line:
            n_water = int(line.split()[-1])
            break

    return {
        "success": sol_result["success"],
        "solvated_gro": str(solvated_gro),
        "topology": top_path,
        "water_molecules_added": n_water,
        "water_model": water_model,
        "error": sol_result["stderr"] if not sol_result["success"] else None,
    }


# ---------------------------------------------------------------------------
# Simulation stages
# ---------------------------------------------------------------------------


@mcp.tool()
def energy_minimize(
    gro_path: str,
    top_path: str,
    output_dir: str,
    max_steps: int = 50000,
    maxwarn: int = 1,
) -> dict:
    """
    Run steepest-descent energy minimization.

    Writes em.mdp, runs gmx grompp + gmx mdrun, returns minimized structure.

    Args:
        gro_path: Input structure — typically the solvated .gro.
        top_path: Topology (.top).
        output_dir: Directory for all output files.
        max_steps: Maximum minimization steps (default 50000).
        maxwarn: GROMACS warnings to allow before aborting (default 1).

    Returns:
        Paths to minimized .gro, .log, and .edr; convergence flag.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mdp = outdir / "em.mdp"
    mdp.write_text(templates.render_em(max_steps=max_steps))

    tpr = outdir / "em.tpr"
    pp = gmx.grompp(mdp=mdp, gro=gro_path, top=top_path, output=tpr, maxwarn=maxwarn)
    if not pp["success"]:
        return {"success": False, "error": pp["stderr"], "step": "grompp"}

    run = gmx.mdrun(tpr=tpr, output_prefix=outdir / "em")
    combined = run["stdout"] + run["stderr"]
    converged = "converged" in combined.lower()

    return {
        "success": run["success"],
        "minimized_gro": str(outdir / "em.gro"),
        "log": str(outdir / "em.log"),
        "edr": str(outdir / "em.edr"),
        "converged": converged,
        "error": run["stderr"] if not run["success"] else None,
    }


@mcp.tool()
def equilibrate_nvt(
    gro_path: str,
    top_path: str,
    output_dir: str,
    duration_ps: float = 100.0,
    temperature_K: float = 300.0,
    tc_groups: list[str] | None = None,
    maxwarn: int = 1,
) -> dict:
    """
    NVT equilibration with V-rescale thermostat at constant volume.

    Bring the system to the target temperature before applying pressure coupling.

    Args:
        gro_path: Input structure — typically the energy-minimized .gro.
        top_path: Topology (.top).
        output_dir: Directory for output files.
        duration_ps: Simulation length in picoseconds (default 100).
        temperature_K: Target temperature in Kelvin (default 300).
        tc_groups: Temperature coupling groups, e.g. ["BICH", "SOL"].
                   Requires a matching index file if non-default. Defaults to ["System"].
        maxwarn: GROMACS warnings to allow.

    Returns:
        Paths to equilibrated .gro, .xtc trajectory, .log, and .edr.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mdp = outdir / "nvt.mdp"
    mdp.write_text(
        templates.render_nvt(
            duration_ps=duration_ps,
            temperature=temperature_K,
            tc_groups=tc_groups,
        )
    )

    tpr = outdir / "nvt.tpr"
    pp = gmx.grompp(mdp=mdp, gro=gro_path, top=top_path, output=tpr, maxwarn=maxwarn)
    if not pp["success"]:
        return {"success": False, "error": pp["stderr"], "step": "grompp"}

    run = gmx.mdrun(tpr=tpr, output_prefix=outdir / "nvt")

    return {
        "success": run["success"],
        "equilibrated_gro": str(outdir / "nvt.gro"),
        "trajectory": str(outdir / "nvt.xtc"),
        "log": str(outdir / "nvt.log"),
        "edr": str(outdir / "nvt.edr"),
        "duration_ps": duration_ps,
        "temperature_K": temperature_K,
        "error": run["stderr"] if not run["success"] else None,
    }


@mcp.tool()
def equilibrate_npt(
    gro_path: str,
    top_path: str,
    output_dir: str,
    duration_ps: float = 100.0,
    temperature_K: float = 300.0,
    pressure_bar: float = 1.0,
    tc_groups: list[str] | None = None,
    maxwarn: int = 1,
) -> dict:
    """
    NPT equilibration with C-rescale barostat (semiisotropic — surface XY fixed, Z free).

    Equilibrates box size and density before production. Requires a prior NVT run.

    Args:
        gro_path: Input structure — typically the NVT-equilibrated .gro.
        top_path: Topology (.top).
        output_dir: Directory for output files.
        duration_ps: Simulation length in picoseconds (default 100).
        temperature_K: Target temperature in Kelvin (default 300).
        pressure_bar: Target pressure in bar (default 1.0).
        tc_groups: Temperature coupling groups (default ["System"]).
        maxwarn: GROMACS warnings to allow.

    Returns:
        Paths to equilibrated .gro, .xtc trajectory, .log, and .edr.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mdp = outdir / "npt.mdp"
    mdp.write_text(
        templates.render_npt(
            duration_ps=duration_ps,
            temperature=temperature_K,
            tc_groups=tc_groups,
        )
    )

    tpr = outdir / "npt.tpr"
    pp = gmx.grompp(mdp=mdp, gro=gro_path, top=top_path, output=tpr, maxwarn=maxwarn)
    if not pp["success"]:
        return {"success": False, "error": pp["stderr"], "step": "grompp"}

    run = gmx.mdrun(tpr=tpr, output_prefix=outdir / "npt")

    return {
        "success": run["success"],
        "equilibrated_gro": str(outdir / "npt.gro"),
        "trajectory": str(outdir / "npt.xtc"),
        "log": str(outdir / "npt.log"),
        "edr": str(outdir / "npt.edr"),
        "duration_ps": duration_ps,
        "temperature_K": temperature_K,
        "pressure_bar": pressure_bar,
        "error": run["stderr"] if not run["success"] else None,
    }


@mcp.tool()
def run_production(
    gro_path: str,
    top_path: str,
    output_dir: str,
    duration_ps: float = 300.0,
    temperature_K: float = 300.0,
    pressure_bar: float = 1.0,
    tc_groups: list[str] | None = None,
    maxwarn: int = 1,
) -> dict:
    """
    Production MD with Nosé-Hoover thermostat and Parrinello-Rahman barostat.

    Generates the trajectory used for analysis. Semiisotropic pressure coupling
    keeps the surface XY dimensions fixed while the Z-axis (solvent layer) relaxes.

    Args:
        gro_path: Input structure — typically the NPT-equilibrated .gro.
        top_path: Topology (.top).
        output_dir: Directory for output files.
        duration_ps: Simulation length in picoseconds (default 300).
        temperature_K: Target temperature in Kelvin (default 300).
        pressure_bar: Target pressure in bar (default 1.0).
        tc_groups: Temperature coupling groups (default ["System"]).
        maxwarn: GROMACS warnings to allow.

    Returns:
        Paths to final .gro, .xtc trajectory, .tpr, .log, .edr, and checkpoint.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mdp = outdir / "production.mdp"
    mdp.write_text(
        templates.render_production(
            duration_ps=duration_ps,
            temperature=temperature_K,
            tc_groups=tc_groups,
        )
    )

    tpr = outdir / "production.tpr"
    pp = gmx.grompp(mdp=mdp, gro=gro_path, top=top_path, output=tpr, maxwarn=maxwarn)
    if not pp["success"]:
        return {"success": False, "error": pp["stderr"], "step": "grompp"}

    run = gmx.mdrun(tpr=tpr, output_prefix=outdir / "production")

    return {
        "success": run["success"],
        "final_gro": str(outdir / "production.gro"),
        "trajectory": str(outdir / "production.xtc"),
        "tpr": str(tpr),
        "log": str(outdir / "production.log"),
        "edr": str(outdir / "production.edr"),
        "checkpoint": str(outdir / "production.cpt"),
        "duration_ps": duration_ps,
        "error": run["stderr"] if not run["success"] else None,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_energy(edr_path: str) -> dict:
    """
    Extract thermodynamic statistics from a GROMACS energy file (.edr).

    Args:
        edr_path: Path to .edr file from any simulation stage.

    Returns:
        Mean, std, min, max for potential energy (kJ/mol), temperature (K),
        pressure (bar), and density (kg/m³).
    """
    return analysis.energy_summary(edr_path)


@mcp.tool()
def analyze_density_profile(
    tpr_path: str,
    xtc_path: str,
    output_dir: str,
    selection: str = "resname SOL and name OW",
    start_frame: int = 0,
) -> dict:
    """
    Compute z-axis mass density profile from a production trajectory.

    Reveals water layering and depletion zones near the biochar surface.

    Args:
        tpr_path: GROMACS run input file (.tpr) from the production run.
        xtc_path: Compressed trajectory (.xtc) from the production run.
        output_dir: Directory to write density_profile.json.
        selection: MDAnalysis atom selection for density (default: water oxygen).
        start_frame: First trajectory frame to include (skip early equilibration).

    Returns:
        z positions (nm) and mass density (kg/m³), saved to density_profile.json.
    """
    result = analysis.density_profile(
        tpr_path=tpr_path,
        xtc_path=xtc_path,
        selection=selection,
        start_frame=start_frame,
    )
    out = Path(output_dir) / "density_profile.json"
    out.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out)
    return result


@mcp.tool()
def analyze_rdf(
    tpr_path: str,
    xtc_path: str,
    output_dir: str,
    selection_a: str = "resname BICH and name CA",
    selection_b: str = "resname SOL and name OW",
    r_max_nm: float = 1.5,
    start_frame: int = 0,
) -> dict:
    """
    Compute radial distribution function (RDF) between two atom selections.

    Default: biochar surface carbons vs. water oxygen atoms.
    The first peak position and height characterize surface hydrophilicity.

    Args:
        tpr_path: GROMACS run input file (.tpr).
        xtc_path: Compressed trajectory (.xtc).
        output_dir: Directory to write rdf.json.
        selection_a: First selection — typically the surface atoms.
        selection_b: Second selection — typically the solvent probe atoms.
        r_max_nm: Maximum RDF distance in nm (default 1.5).
        start_frame: First trajectory frame to include.

    Returns:
        r (nm) and g(r) arrays, saved to rdf.json.
    """
    result = analysis.radial_distribution_function(
        tpr_path=tpr_path,
        xtc_path=xtc_path,
        selection_a=selection_a,
        selection_b=selection_b,
        r_max_nm=r_max_nm,
        start_frame=start_frame,
    )
    out = Path(output_dir) / "rdf.json"
    out.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out)
    return result


@mcp.tool()
def analyze_sasa(
    tpr_path: str,
    xtc_path: str,
    output_dir: str,
    probe_radius_nm: float = 0.14,
    start_frame: int = 0,
) -> dict:
    """
    Compute solvent-accessible surface area (SASA) of the biochar surface via gmx sasa.

    Args:
        tpr_path: GROMACS run input file (.tpr).
        xtc_path: Compressed trajectory (.xtc).
        output_dir: Directory for sasa.xvg output and results.
        probe_radius_nm: Probe radius in nm (default 0.14 — water molecule radius).
        start_frame: First trajectory frame to include.

    Returns:
        Mean and std SASA (nm²), per-frame time series.
    """
    return analysis.solvent_accessible_surface_area(
        tpr_path=tpr_path,
        xtc_path=xtc_path,
        output_dir=output_dir,
        probe_radius_nm=probe_radius_nm,
        start_frame=start_frame,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
