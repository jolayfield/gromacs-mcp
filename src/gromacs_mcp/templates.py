"""MDP parameter templates for each GROMACS simulation stage.

All templates use modern GROMACS defaults (2021+):
  - Verlet cutoff scheme
  - PME electrostatics with 1.2 nm cutoff
  - LINCS constraints on H-bonds (2 fs timestep)
  - DispCorr = EnerPres for long-range LJ correction

Pressure coupling style is selected at render time:
  - "semiisotropic": XY fixed, Z free — for surface/slab simulations
  - "isotropic": all axes coupled — for peptide/protein in solution
"""
from __future__ import annotations

_EM = """\
; Energy Minimization — steepest descent
integrator      = steep
nsteps          = {max_steps}
emtol           = 1000.0
emstep          = 0.01

nstxout-compressed  = 500
nstlog              = 500

cutoff-scheme   = Verlet
nstlist         = 20
pbc             = xyz
rcoulomb        = 1.2
rvdw            = 1.2
rvdw-switch     = 1.0
rlist           = 1.2
coulombtype     = PME
pme-order       = 4
fourierspacing  = 0.16
DispCorr        = EnerPres
"""

_NVT = """\
; NVT Equilibration — V-rescale thermostat
integrator      = md
nsteps          = {nsteps}
dt              = 0.002

nstxout-compressed  = 500
nstlog              = 500
nstcalcenergy       = 100
nstenergy           = 500

cutoff-scheme   = Verlet
nstlist         = 20
pbc             = xyz
rcoulomb        = 1.2
rvdw            = 1.2
rvdw-switch     = 1.0
rlist           = 1.2
coulombtype     = PME
pme-order       = 4
fourierspacing  = 0.16
DispCorr        = EnerPres

tcoupl          = V-rescale
tc-grps         = {tc_grps}
tau_t           = {tau_t}
ref_t           = {ref_t}
pcoupl          = no

constraints             = h-bonds
constraint-algorithm    = lincs
lincs-iter              = 1
lincs-order             = 4

gen_vel         = yes
gen_temp        = {temperature}
gen_seed        = -1

comm-mode       = Linear
nstcomm         = 100
"""

_NPT = """\
; NPT Equilibration — C-rescale barostat (GROMACS 2021+)
integrator      = md
nsteps          = {nsteps}
dt              = 0.002

nstxout-compressed  = 500
nstlog              = 500
nstcalcenergy       = 100
nstenergy           = 500

cutoff-scheme   = Verlet
nstlist         = 20
pbc             = xyz
rcoulomb        = 1.2
rvdw            = 1.2
rvdw-switch     = 1.0
rlist           = 1.2
coulombtype     = PME
pme-order       = 4
fourierspacing  = 0.16
DispCorr        = EnerPres

tcoupl          = V-rescale
tc-grps         = {tc_grps}
tau_t           = {tau_t}
ref_t           = {ref_t}

{pressure_block}

constraints             = h-bonds
constraint-algorithm    = lincs
lincs-iter              = 1
lincs-order             = 4

gen_vel         = no

comm-mode       = Linear
nstcomm         = 100
"""

_PRODUCTION = """\
; Production MD — Nosé-Hoover + Parrinello-Rahman (correct NPT ensemble)
integrator      = md
nsteps          = {nsteps}
dt              = 0.002

nstxout-compressed  = 500
nstlog              = 500
nstcalcenergy       = 100
nstenergy           = 500

cutoff-scheme   = Verlet
nstlist         = 20
pbc             = xyz
rcoulomb        = 1.2
rvdw            = 1.2
rvdw-switch     = 1.0
rlist           = 1.2
coulombtype     = PME
pme-order       = 4
fourierspacing  = 0.16
DispCorr        = EnerPres

tcoupl          = Nose-Hoover
tc-grps         = {tc_grps}
tau_t           = {tau_t}
ref_t           = {ref_t}

{pressure_block}

constraints             = h-bonds
constraint-algorithm    = lincs
lincs-iter              = 1
lincs-order             = 4

gen_vel         = no

comm-mode       = Linear
nstcomm         = 100
"""

# ---------------------------------------------------------------------------
# Pressure coupling blocks
# ---------------------------------------------------------------------------

_PRESSURE_NPT_SEMIISOTROPIC = """\
pcoupl          = C-rescale
pcoupltype      = semiisotropic
tau_p           = 2.0
ref_p           = 1.0   1.0
compressibility = 0     4.5e-5
refcoord-scaling = com"""

_PRESSURE_NPT_ISOTROPIC = """\
pcoupl          = C-rescale
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = 1.0
compressibility = 4.5e-5
refcoord-scaling = com"""

_PRESSURE_PROD_SEMIISOTROPIC = """\
pcoupl          = Parrinello-Rahman
pcoupltype      = semiisotropic
tau_p           = 2.0
ref_p           = 1.0   1.0
compressibility = 0     4.5e-5
refcoord-scaling = com"""

_PRESSURE_PROD_ISOTROPIC = """\
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = 1.0
compressibility = 4.5e-5
refcoord-scaling = com"""


def _pressure_block_npt(coupling: str, pressure_bar: float = 1.0) -> str:
    if coupling == "isotropic":
        return (
            f"pcoupl          = C-rescale\n"
            f"pcoupltype      = isotropic\n"
            f"tau_p           = 2.0\n"
            f"ref_p           = {pressure_bar}\n"
            f"compressibility = 4.5e-5\n"
            f"refcoord-scaling = com"
        )
    return (
        f"pcoupl          = C-rescale\n"
        f"pcoupltype      = semiisotropic\n"
        f"tau_p           = 2.0\n"
        f"ref_p           = 1.0   {pressure_bar}\n"
        f"compressibility = 0     4.5e-5\n"
        f"refcoord-scaling = com"
    )


def _pressure_block_production(coupling: str, pressure_bar: float = 1.0) -> str:
    if coupling == "isotropic":
        return (
            f"pcoupl          = Parrinello-Rahman\n"
            f"pcoupltype      = isotropic\n"
            f"tau_p           = 2.0\n"
            f"ref_p           = {pressure_bar}\n"
            f"compressibility = 4.5e-5\n"
            f"refcoord-scaling = com"
        )
    return (
        f"pcoupl          = Parrinello-Rahman\n"
        f"pcoupltype      = semiisotropic\n"
        f"tau_p           = 2.0\n"
        f"ref_p           = 1.0   {pressure_bar}\n"
        f"compressibility = 0     4.5e-5\n"
        f"refcoord-scaling = com"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc_args(temperature: float, groups: list[str]) -> tuple[str, str, str]:
    """Return (tc_grps, tau_t, ref_t) strings for the given groups."""
    n = len(groups)
    return (
        " ".join(groups),
        " ".join(["0.1"] * n),
        " ".join([str(temperature)] * n),
    )


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


def render_em(max_steps: int = 50000) -> str:
    return _EM.format(max_steps=max_steps)


def render_nvt(
    duration_ps: float = 100.0,
    temperature: float = 300.0,
    tc_groups: list[str] | None = None,
) -> str:
    groups = tc_groups or ["System"]
    tc_grps, tau_t, ref_t = _tc_args(temperature, groups)
    return _NVT.format(
        nsteps=int(duration_ps / 0.002),
        temperature=temperature,
        tc_grps=tc_grps,
        tau_t=tau_t,
        ref_t=ref_t,
    )


def render_npt(
    duration_ps: float = 100.0,
    temperature: float = 300.0,
    tc_groups: list[str] | None = None,
    pressure_coupling: str = "semiisotropic",
    pressure_bar: float = 1.0,
) -> str:
    groups = tc_groups or ["System"]
    tc_grps, tau_t, ref_t = _tc_args(temperature, groups)
    return _NPT.format(
        nsteps=int(duration_ps / 0.002),
        tc_grps=tc_grps,
        tau_t=tau_t,
        ref_t=ref_t,
        pressure_block=_pressure_block_npt(pressure_coupling, pressure_bar),
    )


def render_production(
    duration_ps: float = 300.0,
    temperature: float = 300.0,
    tc_groups: list[str] | None = None,
    pressure_coupling: str = "semiisotropic",
    pressure_bar: float = 1.0,
) -> str:
    groups = tc_groups or ["System"]
    tc_grps, tau_t, ref_t = _tc_args(temperature, groups)
    return _PRODUCTION.format(
        nsteps=int(duration_ps / 0.002),
        tc_grps=tc_grps,
        tau_t=tau_t,
        ref_t=ref_t,
        pressure_block=_pressure_block_production(pressure_coupling, pressure_bar),
    )
