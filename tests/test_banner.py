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

    # Use the helper to derive geometry for an 80-column terminal
    geom = banner._globe_geometry(80)  # (space_width, cx, cy, rx, ry)

    scene_0 = banner._render_planet_globe(0.0, -1.0, *geom)
    scene_pi = banner._render_planet_globe(math.pi, -1.0, *geom)

    assert len(scene_0) == banner.SPACE_ROWS
    assert len(scene_pi) == banner.SPACE_ROWS
    # Different rotation angle produces different surface shading/texture
    assert scene_0 != scene_pi

    # Transit planet with phase 0.5 should place the disc near the centre
    scene_transit = banner._render_planet_globe(0.0, 0.5, *geom)
    assert len(scene_transit) == banner.SPACE_ROWS
    assert scene_transit != scene_0, "transit-bearing frame must differ from clear-globe frame"


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


def test_prebaked_frames_contract():
    """The baked matrix is exactly 32 frames of 19 rows x 56 visible columns."""
    banner = _import_banner()
    frames = banner._PREBAKED_GLOBE_FRAMES

    assert len(frames) == 32
    for frame in frames:
        assert isinstance(frame, tuple)
        assert len(frame) == banner.SPACE_ROWS
        for row in frame:
            assert len(banner._strip_ansi(row)) == 56

    # Deterministic bake: re-rendering any frame must reproduce it byte-for-byte.
    geom = banner._globe_geometry(56)
    rebuilt = banner._render_planet_globe(2.0 * math.pi * 5 / 32, -1.0, *geom)
    assert tuple(rebuilt) == frames[5]

    # Transit planet sweeps only across frames 12..28 inclusive.
    transit_marker = banner._TRANSIT_BG
    assert all(transit_marker not in row for row in frames[0])
    assert any(transit_marker in row for row in frames[12])
    assert any(transit_marker in row for row in frames[20])
    assert any(transit_marker in row for row in frames[28])
    assert all(
        transit_marker not in row
        for index, frame in enumerate(frames)
        if index not in range(12, 29)
        for row in frame
    )


def test_run_banner_streams_prebaked_frames_without_rerendering(monkeypatch):
    """The streaming loop must read the baked matrix, never call the renderer."""
    banner = _import_banner()
    calls = {"render": 0}
    real_render = banner._render_planet_globe

    def counting_render(*args, **kwargs):
        calls["render"] += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(banner, "_render_planet_globe", counting_render)
    monkeypatch.setattr(banner.time, "sleep", lambda _seconds: None)

    fake_stdout = io.StringIO()
    with patch("sys.stdout", fake_stdout):
        with patch.object(fake_stdout, "isatty", return_value=True):
            with patch.object(io, "StringIO", return_value=fake_stdout):
                # get_terminal_size patched at os level used inside module.
                with patch("os.get_terminal_size", lambda: __import__("os").terminal_size((100, 30))):
                    # Force an immediate keypress exit after the grace period.
                    monkeypatch.setattr(banner, "_check_user_key", lambda: True)
                    banner.run_banner()

    assert calls["render"] == 0, "run_banner must stream pre-baked frames without raytracing"
    out = fake_stdout.getvalue()
    assert "\x1b[?25h" in out, "cursor must be restored"

