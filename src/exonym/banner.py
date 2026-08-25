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
# Terminal character cells are roughly twice as tall as they are wide (~2.1:1).
# A circular sphere needs radius_x ≈ radius_y * 2.2.
# In a 19-row vertical canvas with center_y = 9, radius_y = 7.5 guarantees the
# full sphere (including atmosphere glow) fits perfectly without pole clipping.

GLOBE_RADIUS_Y_MAX: float = 7.5
GLOBE_RADIUS_X_MAX: float = 17.0
_GLOBE_ASPECT: float = GLOBE_RADIUS_X_MAX / GLOBE_RADIUS_Y_MAX

# Fixed vertical geometry (rows) -- designed to fit on standard 24+ row terminals
SPACE_ROWS: int = 19

# Background space stars as (x_frac, y_frac, glyph) -- scaled to canvas
_SPACE_STARS_FRAC: Tuple[Tuple[float, float, str], ...] = (
    (0.047, 0.053, "."),
    (0.109, 0.789, "*"),
    (0.875, 0.105, "+"),
    (0.937, 0.579, "."),
    (0.203, 0.895, "·"),
    (0.797, 0.842, "·"),
    (0.031, 0.474, "·"),
    (0.953, 0.316, "*"),
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
# gap line before the globe.
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

# ---------------------------------------------------------------------------
# Transit planet — fully opaque pitch-black silhouette disc
# ---------------------------------------------------------------------------
_TRANSIT_RADIUS_X: float = 3.5
_TRANSIT_RADIUS_Y: float = 1.75
_TRANSIT_CHAR: str = " "
_TRANSIT_BG: str = _ESC + "[40m"
_TRANSIT_COLOR: str = _fg(0)


def _globe_geometry(term_width: int) -> Tuple[int, int, int, float, float]:
    """Return (space_width, center_x, center_y, radius_x, radius_y) for terminal width.

    The canvas spans the available terminal width (up to 120 columns) so the
    starfield expands cleanly across wide PowerShell windows. The globe itself
    is scaled to fit the 19-row vertical canvas with zero clipping at the poles.
    """
    space_width = max(50, min(term_width - 2, 120))
    center_x = space_width // 2
    center_y = SPACE_ROWS // 2

    # Vertical geometry constraint guarantees full spherical shape with no clipping
    radius_y = GLOBE_RADIUS_Y_MAX
    # Horizontal radius maintains 2.2:1 aspect ratio, bounded by canvas width
    max_rx_canvas = max(10.0, (space_width // 2) - 4.0)
    radius_x = min(GLOBE_RADIUS_X_MAX, max_rx_canvas)
    radius_y = radius_x / _GLOBE_ASPECT

    return space_width, center_x, center_y, radius_x, radius_y


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
                msvcrt.getch()
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


def _render_planet_globe(
    angle: float,
    transit_phase: float,
    space_width: int,
    center_x: int,
    center_y: int,
    radius_x: float,
    radius_y: float,
) -> List[str]:
    """Raytrace a high-definition 3D rotating exoplanet globe in space,
    with a small pitch-black disc transiting across its face.

    Args:
        angle: Current rotation angle of the globe in radians.
        transit_phase: 0-1 transit progression (0 = entering left,
            0.5 = centred, 1 = exiting right).  Values outside [0,1]
            mean no transit planet is drawn.
        space_width: Canvas column count (derived from terminal width).
        center_x: Horizontal centre of the globe in canvas columns.
        center_y: Vertical centre of the globe in canvas rows.
        radius_x: Globe X radius in terminal columns.
        radius_y: Globe Y radius in terminal rows.

    Returns:
        Rows of ANSI-coloured space + globe + optional transit planet.
    """
    grid: List[List[str]] = [[" "] * space_width for _ in range(SPACE_ROWS)]

    # Draw space stars (fractional positions -> integer cells across space_width)
    for sx_f, sy_f, sch in _SPACE_STARS_FRAC:
        sx = int(round(sx_f * (space_width - 1)))
        sy = int(round(sy_f * (SPACE_ROWS - 1)))
        if 0 <= sx < space_width and 0 <= sy < SPACE_ROWS:
            grid[sy][sx] = _GREY + sch + _RESET

    # Directional lighting vector
    lx, ly, lz = -0.5, -0.4, 0.76
    norm_l = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / norm_l, ly / norm_l, lz / norm_l

    # Raytrace each cell in the sphere; loop bounds cover the atmosphere glow.
    x_span = int(math.ceil(radius_x * 1.15)) + 1
    y_span = int(math.ceil(radius_y * 1.15)) + 1
    for y_idx in range(-y_span, y_span + 1):
        r_row = center_y + y_idx
        for x_idx in range(-x_span, x_span + 1):
            r_col = center_x + x_idx
            if 0 <= r_row < SPACE_ROWS and 0 <= r_col < space_width:
                nx = x_idx / radius_x
                ny = y_idx / radius_y
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

    # --- Draw the transiting planet (pure black opaque silhouette disc) ---
    transit_rx = _TRANSIT_RADIUS_X
    transit_ry = _TRANSIT_RADIUS_Y
    transit_margin = transit_rx + 2.0
    if 0.0 <= transit_phase <= 1.0:
        travel_left = center_x - radius_x - transit_margin
        travel_right = center_x + radius_x + transit_margin
        planet_cx = travel_left + transit_phase * (travel_right - travel_left)
        planet_cy = center_y

        px_span = int(math.ceil(transit_rx * 1.1))
        py_span = int(math.ceil(transit_ry * 1.1))
        for dy in range(-py_span, py_span + 1):
            py = int(round(planet_cy + dy))
            if not (0 <= py < SPACE_ROWS):
                continue
            for dx in range(-px_span, px_span + 1):
                px = int(round(planet_cx + dx))
                if not (0 <= px < space_width):
                    continue
                cx_f = (px - planet_cx) / transit_rx
                cy_f = (py - planet_cy) / transit_ry
                d2 = cx_f * cx_f + cy_f * cy_f
                if d2 <= 1.0:
                    # Pure black: black background + black foreground space
                    grid[py][px] = _TRANSIT_COLOR + _TRANSIT_BG + _TRANSIT_CHAR + _RESET

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


def _compose_full_screen(angle: float, transit_phase: float, term_width: int) -> str:
    """Compose the full dynamically centered banner frame."""
    space_width, center_x, center_y, radius_x, radius_y = _globe_geometry(term_width)

    lines: List[str] = []

    # 1. Centered Logo with version tagline bridge
    lines.append("")
    for logo_line in _LOGO_LINES:
        formatted = _WHITE + _BOLD + logo_line + _RESET
        lines.append(_center_line(formatted, term_width) + _ESC + "[K")
    lines.append(_center_line(_tagline_line(), term_width) + _ESC + "[K")

    # 2. Centered High-Definition Rotating Exoplanet Globe + Transit in Space
    lines.append(_ESC + "[K")
    for g_line in _render_planet_globe(
        angle, transit_phase,
        space_width, center_x, center_y, radius_x, radius_y,
    ):
        lines.append(_center_line(g_line, term_width) + _ESC + "[K")

    return "\n".join(lines)


def run_banner(skip: bool = False) -> None:
    """Run the high-definition 3D rotating exoplanet globe animation banner.

    Frames are centered horizontally and vertically. The animation adjusts
    dynamically to the terminal width so the planet remains a fully formed
    sphere without distortion or clipping.

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
        if ts.lines < _FRAME_MIN_ROWS:
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
        transit_phase = -1.0       # < 0 -> no planet; starts invisible
        frame_idx = 0
        # Allow at least 12 frames (~0.5s) before checking keys
        while not _interrupted[0]:
            if frame_idx > 12 and _check_user_key():
                break

            try:
                ts = os.get_terminal_size()
                term_width = max(60, ts.columns)
                term_rows = ts.lines
                if ts.lines < _FRAME_MIN_ROWS:
                    break
            except OSError:
                pass

            v_pad = max(0, (term_rows - 1 - _FRAME_LINES) // 2)
            frame_str = _compose_full_screen(angle, transit_phase, term_width)
            stdout.write(_ESC + "[H" + "\n" * v_pad + frame_str + "\n")
            stdout.flush()

            angle += 0.04
            frame_idx += 1

            # Transit planet animation: cycle through phases repeatedly
            # Let the globe rotate solo for ~2s before first transit, then transit every ~3.5s
            _transit_frame = frame_idx - 50   # start first transit after ~2s
            _period_frames = 88                # ~3.5 s per transit cycle
            if _transit_frame >= 0:
                cycle_pos = (_transit_frame % _period_frames) / _period_frames
                transit_phase = cycle_pos
            else:
                transit_phase = -1.0

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
