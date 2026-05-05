"""MDAnalysis-based trajectory analysis for surface MD simulations."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def density_profile(
    tpr_path: str,
    xtc_path: str,
    selection: str = "resname SOL and name OW",
    bin_width_angstrom: float = 1.0,
    start_frame: int = 0,
) -> dict:
    """
    Compute mass density profile along the z-axis.

    Returns z positions (nm) and mass density (kg/m³) for the selection.
    Useful for visualizing water layering near a surface.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.lineardensity import LinearDensity

    u = mda.Universe(tpr_path, xtc_path)
    sel = u.select_atoms(selection)

    ld = LinearDensity(sel, grouping="atoms", binsize=bin_width_angstrom)
    ld.run(start=start_frame)

    return {
        "z_nm": (ld.results.z.centers / 10).tolist(),
        "density_kg_m3": ld.results.z.mass_density.tolist(),
        "selection": selection,
        "n_frames": len(u.trajectory) - start_frame,
        "units": {"z": "nm", "density": "kg/m³"},
    }


def radial_distribution_function(
    tpr_path: str,
    xtc_path: str,
    selection_a: str = "resname BICH and name CA",
    selection_b: str = "resname SOL and name OW",
    r_max_nm: float = 1.5,
    n_bins: int = 150,
    start_frame: int = 0,
) -> dict:
    """
    Compute RDF between two atom selections.

    Default: surface carbon atoms vs. water oxygen — characterizes hydrophilicity.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.rdf import InterRDF

    u = mda.Universe(tpr_path, xtc_path)
    a = u.select_atoms(selection_a)
    b = u.select_atoms(selection_b)

    rdf = InterRDF(a, b, nbins=n_bins, range=(0.0, r_max_nm * 10))  # nm → Å
    rdf.run(start=start_frame)

    return {
        "r_nm": (rdf.results.bins / 10).tolist(),
        "g_r": rdf.results.rdf.tolist(),
        "selection_a": selection_a,
        "selection_b": selection_b,
        "units": {"r": "nm", "g_r": "dimensionless"},
    }


def energy_summary(edr_path: str) -> dict:
    """
    Extract thermodynamic statistics from a GROMACS .edr energy file.

    Returns mean/std/min/max for potential energy, temperature, pressure, density.
    """
    import pyedr

    data = pyedr.edr_to_dict(edr_path)

    def _stats(key: str) -> dict | None:
        if key not in data:
            return None
        arr = np.array(data[key])
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "potential_energy_kJ_mol": _stats("Potential"),
        "kinetic_energy_kJ_mol": _stats("Kinetic En."),
        "total_energy_kJ_mol": _stats("Total Energy"),
        "temperature_K": _stats("Temperature"),
        "pressure_bar": _stats("Pressure"),
        "density_kg_m3": _stats("Density"),
    }


def solvent_accessible_surface_area(
    tpr_path: str,
    xtc_path: str,
    output_dir: str,
    probe_radius_nm: float = 0.14,
    start_frame: int = 0,
) -> dict:
    """
    Compute SASA of the biochar surface using gmx sasa.

    Calls the GROMACS built-in sasa tool and parses the XVG output.
    Group 0 = System (surface definition), Group 1 = output group.
    """
    from . import gmx

    outdir = Path(output_dir)
    xvg_path = outdir / "sasa.xvg"

    result = gmx.run(
        "sasa",
        "-s", str(tpr_path),
        "-f", str(xtc_path),
        "-o", str(xvg_path),
        "-probe", str(probe_radius_nm),
        "-b", "0",
        stdin="0\n",
    )

    if not result["success"]:
        return {"success": False, "error": result["stderr"]}

    times, values = [], []
    with open(xvg_path) as fh:
        for line in fh:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                values.append(float(parts[1]))

    arr = np.array(values)
    return {
        "success": True,
        "mean_nm2": float(arr.mean()),
        "std_nm2": float(arr.std()),
        "time_ps": times,
        "sasa_nm2": values,
        "probe_radius_nm": probe_radius_nm,
        "units": {"sasa": "nm²", "time": "ps"},
    }
