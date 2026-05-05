"""Thin subprocess wrapper around GROMACS command-line tools."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_gmx() -> str:
    for name in ("gmx", "gmx_mpi", "gmx_d"):
        if path := shutil.which(name):
            return path
    raise RuntimeError(
        "GROMACS not found in PATH. Install GROMACS and ensure 'gmx' is accessible."
    )


def run(
    *args: str,
    cwd: str | Path | None = None,
    stdin: str | None = None,
) -> dict:
    """Run a gmx subcommand and return a structured result."""
    cmd = [find_gmx()] + list(args)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        input=stdin,
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(cmd),
    }


def grompp(
    mdp: str | Path,
    gro: str | Path,
    top: str | Path,
    output: str | Path,
    ndx: str | Path | None = None,
    maxwarn: int = 0,
    cwd: str | Path | None = None,
) -> dict:
    args = [
        "grompp",
        "-f", str(mdp),
        "-c", str(gro),
        "-p", str(top),
        "-o", str(output),
        "-maxwarn", str(maxwarn),
    ]
    if ndx:
        args += ["-n", str(ndx)]
    return run(*args, cwd=cwd)


def mdrun(
    tpr: str | Path,
    output_prefix: str | Path,
    ntmpi: int = 1,
    ntomp: int = 1,
    gpu_id: str | None = None,
    cwd: str | Path | None = None,
) -> dict:
    args = [
        "mdrun",
        "-s", str(tpr),
        "-deffnm", str(output_prefix),
        "-ntmpi", str(ntmpi),
        "-ntomp", str(ntomp),
    ]
    if gpu_id is not None:
        args += ["-gpu_id", gpu_id]
    return run(*args, cwd=cwd)
