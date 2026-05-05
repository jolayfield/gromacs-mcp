"""MDP parameter templates for each GROMACS simulation stage.

All templates use modern GROMACS defaults (2021+):
  - Verlet cutoff scheme
  - PME electrostatics with 1.2 nm cutoff
  - LINCS constraints on H-bonds (2 fs timestep)
  - DispCorr = EnerPres for long-range LJ correction
  - Semiisotropic pressure coupling for surface simulations
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
; Semiisotropic: XY fixed (surface lattice), Z flexible (solvent layer)
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

pcoupl          = C-rescale
pcoupltype      = semiisotropic
tau_p           = 2.0
ref_p           = 1.0   1.0
compressibility = 0     4.5e-5
refcoord-scaling = com

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
; Semiisotropic: XY fixed, Z flexible
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

pcoupl          = Parrinello-Rahman
pcoupltype      = semiisotropic
tau_p           = 2.0
ref_p           = 1.0   1.0
compressibility = 0     4.5e-5
refcoord-scaling = com

constraints             = h-bonds
constraint-algorithm    = lincs
lincs-iter              = 1
lincs-order             = 4

gen_vel         = no

comm-mode       = Linear
nstcomm         = 100
"""


def _tc_args(temperature: float, groups: list[str]) -> tuple[str, str, str]:
    """Return (tc_grps, tau_t, ref_t) strings for the given groups."""
    n = len(groups)
    return (
        " ".join(groups),
        " ".join(["0.1"] * n),
        " ".join([str(temperature)] * n),
    )


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
) -> str:
    groups = tc_groups or ["System"]
    tc_grps, tau_t, ref_t = _tc_args(temperature, groups)
    return _NPT.format(
        nsteps=int(duration_ps / 0.002),
        tc_grps=tc_grps,
        tau_t=tau_t,
        ref_t=ref_t,
    )


def render_production(
    duration_ps: float = 300.0,
    temperature: float = 300.0,
    tc_groups: list[str] | None = None,
) -> str:
    groups = tc_groups or ["System"]
    tc_grps, tau_t, ref_t = _tc_args(temperature, groups)
    return _PRODUCTION.format(
        nsteps=int(duration_ps / 0.002),
        tc_grps=tc_grps,
        tau_t=tau_t,
        ref_t=ref_t,
    )
