"""Pytest configuration and shared fixtures."""
import shutil

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_gromacs: test requires a GROMACS installation in PATH",
    )


@pytest.fixture(autouse=True)
def skip_without_gromacs(request):
    if request.node.get_closest_marker("requires_gromacs"):
        if not shutil.which("gmx"):
            pytest.skip("GROMACS not found in PATH")
