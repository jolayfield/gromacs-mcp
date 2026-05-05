# gromacs-mcp

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that exposes the GROMACS molecular dynamics pipeline as Claude-callable tools. It lets Claude Code drive full MD simulations — from solvation through production and analysis — without manual terminal work.

## Tools

| Tool | Description |
|------|-------------|
| `check_gromacs` | Verify GROMACS is installed and return its version |
| `solvate` | Add a periodic water box around a dry structure (`gmx editconf` + `gmx solvate`) |
| `make_index` | Create a custom index file with named atom groups (`gmx make_ndx`) |
| `energy_minimize` | Steepest-descent energy minimization (`gmx grompp` + `gmx mdrun`) |
| `equilibrate_nvt` | NVT equilibration with V-rescale thermostat |
| `equilibrate_npt` | NPT equilibration with C-rescale barostat (semiisotropic) |
| `run_production` | Production MD with Nosé-Hoover + Parrinello-Rahman |
| `analyze_energy` | Extract thermodynamic statistics from an `.edr` file |
| `analyze_rdf` | Radial distribution function between two atom selections |
| `analyze_density_profile` | z-axis mass density profile from a trajectory |
| `analyze_sasa` | Solvent-accessible surface area via `gmx sasa` |

## Standard Workflow

```
solvate → energy_minimize → equilibrate_nvt → equilibrate_npt → run_production → analyze_*
```

Each tool returns file paths that feed directly into the next step. For slab-geometry systems (e.g. biochar/water interfaces), run a dry equilibration first:

```
[dry] energy_minimize → equilibrate_nvt → equilibrate_npt
         ↓ solvate
[wet] energy_minimize → equilibrate_nvt → equilibrate_npt → run_production
```

## Requirements

- GROMACS 2021+ installed and available on `PATH` (or set `GMX` env variable)
- Python 3.11+

## Installation

```bash
pip install gromacs-mcp
```

Or from source (editable):

```bash
git clone https://github.com/jolayfield/gromacs-mcp
cd gromacs-mcp
pip install -e ".[dev]"
```

## Connecting to Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "gromacs-mcp": {
      "command": "gromacs-mcp"
    }
  }
}
```

Or with a specific Python environment:

```json
{
  "mcpServers": {
    "gromacs-mcp": {
      "command": "/path/to/your/env/bin/gromacs-mcp"
    }
  }
}
```

Restart Claude Code and the tools will appear automatically.

## Usage Examples

### Basic solvation + minimization

Ask Claude:
> "Solvate my biochar structure at `sim/biochar.gro` with topology `sim/biochar.top`, minimize it, then run 200 ps NVT at 300 K."

Claude will call `solvate` → `energy_minimize` → `equilibrate_nvt` in sequence, passing output paths between steps.

### Custom temperature coupling groups

```python
# Claude calls make_index first, then passes ndx_path to equilibrate_nvt
make_index(
    gro_path="solvated.gro",
    output_dir="sim/",
    groups=[
        {"residue": "BC",  "name": "Biochar"},
        {"residue": "SOL", "name": "Water"},
    ]
)
# → returns ndx_path, use it in subsequent steps with tc_groups=["Biochar", "Water"]
```

### Analysis

```python
analyze_energy(edr_path="production/production.edr")
# → {"potential_energy": {"mean": -3.2e6, "std": 1200, ...}, "temperature": {...}, ...}

analyze_rdf(
    tpr_path="production/production.tpr",
    xtc_path="production/production.xtc",
    output_dir="analysis/",
    selection_a="resname BC and name C*",
    selection_b="resname SOL and name OW",
)
# → {"r_nm": [...], "g_r": [...], "saved_to": "analysis/rdf.json"}
```

## MDP Templates

Default MDP parameters are in `src/gromacs_mcp/templates.py`. Key settings:

| Stage | Thermostat | Barostat | Default duration |
|-------|-----------|---------|-----------------|
| EM | — | — | 50,000 steps |
| NVT | V-rescale (τ = 0.1 ps) | none | 100 ps |
| NPT | V-rescale (τ = 0.1 ps) | C-rescale semiisotropic | 100 ps |
| Production | Nosé-Hoover (τ = 0.5 ps) | Parrinello-Rahman semiisotropic | 300 ps |

All durations, temperatures, and pressures are overridable via tool arguments.

## Development

```bash
pip install -e ".[dev]"
pytest                          # unit tests (no GROMACS needed)
pytest -m requires_gromacs      # integration tests (GROMACS must be on PATH)
ruff check src/
```

## License

MIT
