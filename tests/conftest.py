"""Shared pytest fixtures for the EXONYM test suite."""

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Automatically set EXOPLANET_REPO_ROOT to the repository root.

    The test ``test_self_check_of_actual_repository`` requires this env var
    to locate the live repository tree and run the isolation audit against it.
    It is set here so CI environments that do not set it explicitly still
    execute the self-check.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if "EXOPLANET_REPO_ROOT" not in os.environ:
        os.environ["EXOPLANET_REPO_ROOT"] = str(repo_root)
