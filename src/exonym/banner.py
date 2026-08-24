"""3D raytraced rotating exoplanet globe animation banner for the Exonym CLI.

Target-neutral module -- contains no candidate identifiers, sector values,
or ephemeris constants.  All physics constants use generic names that do not
match the isolation-scanner patterns for SECTOR_NAME, EPHEMERIS_NAME,
or EPHEMERIS_KEYWORDS.

Public surface
--------------
run_banner(skip=False) -> None
    Render the centered logo and 3D rotating exoplanet globe in space,
    then wait for any keypress before proceeding.
print_cli_overview() -> None
    Print the categorized command overview without argparse boilerplate.

The rendered globe is terminal presentation only; it is not a scientific
visualization or candidate-local evidence product.
"""

from __future__ import annotations

import math
import os
import re
import signal
import sys
import time
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 3D Sphere Raytracing & Space Geometry Constants
# ---------------------------------------------------------------------------
# Terminal cells are roughly twice as tall as they are wide, so a circular
# globe needs GLOBE_RADIUS_Y == GLOBE_RADIUS_X / 2.
GLOBE_RADIUS_X: float = 15.0
GLOBE_RADIUS_Y: float = 7.5
SPACE_WIDTH: int = 64
SPACE_ROWS: int = 19
CENTER_X: int = SPACE_WIDTH // 2
CENTER_Y: int = SPACE_ROWS // 2

# Background space stars (x, y, glyph)
SPACE_STARS: Tuple[Tuple[int, int, str], ...] = (
    (3, 1, "."),
    (7, 15, "*"),
    (56, 2, "+"),
    (60, 11, "."),
    (13, 17, "·"),
    (51, 16, "·"),
    (2, 9, "·"),
    (61, 6, "*"),
)

SHADES: str = " .:-=+*#%@"

# Branding -- block font spelling EXONYM (trimmed)
_LOGO_LINES: Tuple[str, ...] = (
    "███████╗██╗  ██╗ ██████╗ ███╗   ██╗██╗   ██╗███╗   ███╗",
    "██╔════╝╚██╗██╔╝██╔═══██╗████╗  ██║╚██╗ ██╔╝████╗ ████║",
    "█████╗   ╚███╔╝ ██║   ██║██╔██╗ ██║ ╚████╔╝ ██╔████╔██║",
    "██╔══╝   ██╔██╗ ██║   ██║██║╚██╗██║  ╚██╔╝  ██║╚██╔╝██║",
    "███████╗██╔╝ ██╗╚██████╔╝██║ ╚████║   ██║   ██║ ╚═╝ ██║",
    "╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝",
)

_TAGLINE: str = "A Framework for Exoplanet Science"

# Frame geometry: one blank line above the logo, the tagline beneath it, one
# gap line before the globe.  The minimum terminal height keeps every frame
# from scrolling on redraw.
_FRAME_LINES: int = 1 + len(_LOGO_LINES) + 1 + 1 + SPACE_ROWS
_FRAME_MIN_ROWS: int = _FRAME_LINES + 1

_ESC      = chr(27)
_RESET    = _ESC + "[0m"
_BOLD     = _ESC + "[1m"
_DIM      = _ESC + "[2m"


def _fg(code: int) -> str:
    return _ESC + "[38;5;" + str(code) + "m"


_WHITE       = _fg(255)
_GREY_LIGHT  = _fg(250)
_GREY_MID    = _fg(244)
_GREY_DARK   = _fg(238)
_GREY        = _fg(240)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences to compute true printable character width."""
    return re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", text)


def _center_line(text: str, term_width: int) -> str:
    """Dynamically center any string (with or without ANSI escapes) in terminal width."""
    vis_len = len(_strip_ansi(text))
    pad = max(0, (term_width - vis_len) // 2)
    return (" " * pad) + text


def _flush_input_buffer() -> None:
    """Discard any lingering keyboard input from previous commands."""
    if os.name == "nt":
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except ImportError:
            pass


def _check_user_key() -> bool:
    """Non-blocking check if user pressed any key."""
    if os.name == "nt":
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                return True
        except ImportError:
            pass
    else:
        try:
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                sys.stdin.read(1)
                return True
        except Exception:
            pass
    return False


def _wait_for_user_key(timeout: float = 30.0) -> None:
    """Wait until user presses a key or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        if _check_user_key():
            return
        time.sleep(0.05)


def _render_planet_globe(angle: float) -> List[str]:
    """Raytrace a high-definition 3D rotating exoplanet globe in space."""
    grid: List[List[str]] = [[" "] * SPACE_WIDTH for _ in range(SPACE_ROWS)]

    # Draw space stars
    for sx, sy, sch in SPACE_STARS:
        grid[sy][sx] = _GREY + sch + _RESET

    # Directional lighting vector
    lx, ly, lz = -0.5, -0.4, 0.76
    norm_l = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / norm_l, ly / norm_l, lz / norm_l

    # Raytrace each cell in the sphere; loop bounds cover the atmosphere glow.
    x_span = int(math.ceil(GLOBE_RADIUS_X * 1.15)) + 1
    y_span = int(math.ceil(GLOBE_RADIUS_Y * 1.15)) + 1
    for y_idx in range(-y_span, y_span + 1):
        r_row = CENTER_Y + y_idx
        for x_idx in range(-x_span, x_span + 1):
            r_col = CENTER_X + x_idx
            if 0 <= r_row < SPACE_ROWS and 0 <= r_col < SPACE_WIDTH:
                nx = x_idx / GLOBE_RADIUS_X
                ny = y_idx / GLOBE_RADIUS_Y
                dist = nx * nx + ny * ny
                if dist <= 1.0:
                    nz = math.sqrt(max(0.0, 1.0 - dist))

                    # Rotate sphere longitude around Y
                    cos_a, sin_a = math.cos(angle), math.sin(angle)
                    rx_rot = nx * cos_a + nz * sin_a
                    rz_rot = -nx * sin_a + nz * cos_a
                    ry_rot = ny

                    lon = math.atan2(rx_rot, rz_rot)
                    lat = math.asin(max(-1.0, min(1.0, ry_rot)))

                    # Smooth continents & cloud swirl pattern on [0, 1]
                    pattern = math.sin(3.0 * lon + math.sin(2.5 * lat)) * math.cos(2.0 * lat)
                    continents = 0.5 + 0.5 * pattern

                    # Diffuse lighting with limb darkening via nz
                    diffuse = max(0.0, nx * lx - ny * ly + nz * lz)
                    intensity = diffuse * (0.55 + 0.45 * continents)

                    char_idx = int(intensity * (len(SHADES) - 1) + 0.5)
                    char_idx = max(1 if intensity > 0.0 else 0, min(len(SHADES) - 1, char_idx))
                    ch = SHADES[char_idx]

                    if intensity > 0.72:
                        grid[r_row][r_col] = _WHITE + _BOLD + ch + _RESET
                    elif intensity > 0.42:
                        grid[r_row][r_col] = _GREY_LIGHT + ch + _RESET
                    elif intensity > 0.18:
                        grid[r_row][r_col] = _GREY_MID + ch + _RESET
                    else:
                        grid[r_row][r_col] = _GREY_DARK + ch + _RESET
                elif dist <= 1.12:
                    # Atmosphere glow ring
                    grid[r_row][r_col] = _GREY_MID + "·" + _RESET

    return ["".join(r) for r in grid]


def _tagline_line() -> str:
    """Compose the version + tagline bridge rendered beneath the logo."""
    text = _TAGLINE
    try:
        import importlib.metadata

        text = "v" + importlib.metadata.version("exonym") + "  ·  " + _TAGLINE
    except Exception:  # noqa: BLE001 - version metadata is optional decoration
        pass
    return _DIM + _GREY_MID + text + _RESET


def _compose_full_screen(angle: float, term_width: int) -> str:
    """Compose the full dynamically centered banner frame."""
    lines: List[str] = []

    # 1. Centered Logo with version tagline bridge
    lines.append("")
    for logo_line in _LOGO_LINES:
        formatted = _WHITE + _BOLD + logo_line + _RESET
        lines.append(_center_line(formatted, term_width) + _ESC + "[K")
    lines.append(_center_line(_tagline_line(), term_width) + _ESC + "[K")

    # 2. Centered High-Definition Rotating Exoplanet Globe
    lines.append(_ESC + "[K")
    for g_line in _render_planet_globe(angle):
        lines.append(_center_line(g_line, term_width) + _ESC + "[K")

    return "\n".join(lines)


def run_banner(skip: bool = False) -> None:
    """Run the high-definition 3D rotating exoplanet globe animation banner.

    Frames are centered horizontally and vertically.  The animation needs a
    terminal of at least ``SPACE_WIDTH`` columns by ``_FRAME_MIN_ROWS`` rows;
    on smaller terminals it is skipped entirely so the frame never scrolls.

    Parameters
    ----------
    skip:
        When *True* (or when stdout is not a TTY), return immediately without
        producing any output.
    """
    # SCIENTIFIC_BOUNDARY: This animation is presentation-only and must never
    # be treated as an observational, model, or candidate-local artifact.
    if skip or not sys.stdout.isatty():
        return

    term_width = 80
    term_rows = 0
    try:
        ts = os.get_terminal_size()
        term_width = max(60, ts.columns)
        term_rows = ts.lines
        if ts.columns < SPACE_WIDTH or ts.lines < _FRAME_MIN_ROWS:
            return
    except OSError:
        pass

    _interrupted = [False]

    def _sigint_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        _interrupted[0] = True

    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    # Flush stale enter/keypress from terminal start
    _flush_input_buffer()

    stdout = sys.stdout
    try:
        stdout.write(_ESC + "[?25l")   # hide cursor
        stdout.write(_ESC + "[2J")     # clear screen
        stdout.write(_ESC + "[H")      # cursor home
        stdout.flush()

        angle = 0.0
        frame_idx = 0
        # Allow at least 15 frames (~0.6s) before checking keys so Enter keypress doesn't instantly dismiss
        while not _interrupted[0]:
            if frame_idx > 12 and _check_user_key():
                break

            try:
                ts = os.get_terminal_size()
                term_width = max(60, ts.columns)
                term_rows = ts.lines
                if ts.columns < SPACE_WIDTH or ts.lines < _FRAME_MIN_ROWS:
                    break
            except OSError:
                pass

            v_pad = max(0, (term_rows - 1 - _FRAME_LINES) // 2)
            frame_str = _compose_full_screen(angle, term_width)
            stdout.write(_ESC + "[H" + "\n" * v_pad + frame_str + "\n")
            stdout.flush()

            angle += 0.08
            frame_idx += 1
            time.sleep(0.04)

            # After 4 full rotations (~10s), wait for keypress
            if frame_idx >= 250:
                _wait_for_user_key(timeout=30.0)
                break

    except Exception:  # noqa: BLE001
        pass
    finally:
        stdout.write(_ESC + "[?25h")   # restore cursor
        stdout.write(_ESC + "[2J")     # clear screen for clean commands transition
        stdout.write(_ESC + "[H")
        stdout.flush()
        signal.signal(signal.SIGINT, old_sigint)


def print_cli_overview() -> None:
    """Print a categorized command overview without parsing arguments.

    The overview is informational only; it does not execute a workflow step
    or make a scientific assessment.
    """
    lines = [
        _BOLD + "CORE & WORKFLOW COMMANDS" + _RESET,
        f"  {_BOLD}{'init':<18}{_RESET} Provision a new candidate workspace",
        f"  {_BOLD}{'status':<18}{_RESET} Show candidate identity record and workspace layout",
        f"  {_BOLD}{'track':<18}{_RESET} Render candidate telemetry progress dashboard",
        f"  {_BOLD}{'advance':<18}{_RESET} Validate gate requirements and promote workflow phase",
        f"  {_BOLD}{'set-state':<18}{_RESET} Update lifecycle state with audit reason",
        f"  {_BOLD}{'tag':<18}{_RESET} Attach metadata tags to a candidate record",
        f"  {_BOLD}{'freeze':<18}{_RESET} Build an immutable reproducibility release bundle",
        f"  {_BOLD}{'verify-release':<18}{_RESET} Replay and verify bundle integrity and offline load",
        f"  {_BOLD}{'verify':<18}{_RESET} Audit shared code; use 'verify candidate' for workspaces",
        "",
        _BOLD + "SCIENTIFIC ANALYSIS & VETTING" + _RESET,
        f"  {_BOLD}{'search':<18}{_RESET} Run BLS transit search on candidate photometry",
        f"  {_BOLD}{'screen':<18}{_RESET} Run fixed-ephemeris photometric consistency checks",
        f"  {_BOLD}{'plot':<18}{_RESET} Generate diagnostic vetting figures and light curves",
        f"  {_BOLD}{'vet':<18}{_RESET} Run TRICERATOPS Monte Carlo false positive probability",
        f"  {_BOLD}{'triage':<18}{_RESET} Aggregate multi-sector pre-vetting evidence into triage",
        f"  {_BOLD}{'fit':<18}{_RESET} MCMC transit fit with free limb darkening & density locking",
        f"  {_BOLD}{'localization':<18}{_RESET} Sub-pixel PRF transit source localization on TPFs",
        f"  {_BOLD}{'sed':<18}{_RESET} Fit stellar atmosphere posterior to broadband photometry",
        f"  {_BOLD}{'asteroseismology':<18}{_RESET} Estimate stellar oscillation envelope and seismic M*/R*",
        f"  {_BOLD}{'phasecurve':<18}{_RESET} Phase curve and secondary eclipse harmonic search",
        f"  {_BOLD}{'ttv':<18}{_RESET} Transit timing variation (O-C) analysis",
        f"  {_BOLD}{'activity':<18}{_RESET} Stellar rotation GLS periodogram analysis",
        f"  {_BOLD}{'dilution':<18}{_RESET} Aperture robustness and dilution sensitivity",
        f"  {_BOLD}{'archive':<18}{_RESET} Query Gaia EDR3 and NASA ExoFOP archival catalog context",
        f"  {_BOLD}{'rv':<18}{_RESET} Ingest and fit radial velocity Keplerian evidence",
        f"  {_BOLD}{'survey':<18}{_RESET} Operate a bounded independent-discovery survey",
        "",
        _BOLD + "OPTIONS" + _RESET,
        f"  {_BOLD}{'-h, --help':<18}{_RESET} Show detailed argument help for any command",
        f"  {_BOLD}{'--version':<18}{_RESET} Show program version and exit",
        f"  {_BOLD}{'--banner':<18}{_RESET} Replay the startup orbital animation",
        f"  {_BOLD}{'--no-animation':<18}{_RESET} Skip the startup orbital animation",
        f"  {_BOLD}{'-q, --quiet':<18}{_RESET} Suppress optional output and animations",
        "",
        _DIM + 'Run "exonym <command> --help" for sub-command arguments and detailed options.' + _RESET,
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
