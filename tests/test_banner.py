"""Smoke tests for the exonym.banner module.

All tests must pass in non-interactive CI environments (no TTY, no display).
The animation path is never exercised here -- only non-TTY / skip=True paths
and internal physics helpers.
"""

from __future__ import annotations

import importlib
import io
import math
import sys
from unittest.mock import patch

import pytest


def _import_banner():
    """Import (or re-use cached) exonym.banner."""
    return importlib.import_module("exonym.banner")


def test_run_banner_non_tty_no_output():
    """run_banner() must write nothing and return None when stdout is not a TTY."""
    banner = _import_banner()
    fake_stdout = io.StringIO()
    with patch("sys.stdout", fake_stdout):
        with patch.object(fake_stdout, "isatty", return_value=False):
            result = banner.run_banner()
    assert result is None
    assert fake_stdout.getvalue() == "", (
        "run_banner() must not write any output to a non-TTY stdout"
    )


def test_run_banner_skip_no_output(capsys):
    """run_banner(skip=True) must produce no stdout output regardless of TTY."""
    banner = _import_banner()
    with patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = True
        banner.run_banner(skip=True)
        mock_stdout.write.assert_not_called()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_sphere_scene_raytracing():
    """Validate 3D sphere raytracing produces valid character rows and changes with angle."""
    banner = _import_banner()

    scene_0 = banner._render_planet_globe(0.0)
    scene_pi = banner._render_planet_globe(math.pi)

    assert len(scene_0) == banner.SPACE_ROWS
    assert len(scene_pi) == banner.SPACE_ROWS
    # Different rotation angle produces different surface shading/texture
    assert scene_0 != scene_pi


def test_print_cli_overview_output(capsys):
    """print_cli_overview() must output the clean categorized command sections."""
    banner = _import_banner()
    banner.print_cli_overview()
    captured = capsys.readouterr()

    assert "CORE & WORKFLOW COMMANDS" in captured.out
    assert "SCIENTIFIC ANALYSIS & VETTING" in captured.out
    assert "OPTIONS" in captured.out
    assert "init" in captured.out
    assert "vet" in captured.out




def test_banner_import_is_isolated():
    """Importing exonym.banner must not pull in any other exonym submodule."""
    for key in list(sys.modules.keys()):
        if key == "exonym.banner":
            del sys.modules[key]

    pre_modules = {k for k in sys.modules if k.startswith("exonym.")}

    import exonym.banner  # noqa: F401

    post_modules = {k for k in sys.modules if k.startswith("exonym.")}
    new_modules = post_modules - pre_modules - {"exonym.banner"}

    assert not new_modules, (
        "Importing exonym.banner pulled in unexpected exonym submodules: "
        + ", ".join(sorted(new_modules))
    )
