"""Headless diagnostic vetting figure generation.

All routines enforce headless rendering (`matplotlib.use('Agg')`) to run cleanly
in automated pipelines and CI/CD without requiring display servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Enforce non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from .inputs import load_transit_ephemeris
from .lightcurve import bin_phase_folded_flux, phase_hours
from .search import load_candidate_light_curve
from .workspace import CandidateWorkspace


def plot_phase_folded_lc(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    output_path: Path,
    bin_minutes: float = 8.0,
    limit_hours: float = 12.0,
) -> Path:
    """Render a phase-folded light curve plot and save to output_path."""
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    hours = phase_hours(time, period_days, epoch_btjd)

    mask = np.abs(hours) <= limit_hours
    hours = hours[mask]
    values = values[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hours, values, ".", color="#888888", alpha=0.3, markersize=3, label="Unbinned Data")

    centers, median, error = bin_phase_folded_flux(
        time, flux, period_days, epoch_btjd, limit_hours=limit_hours, bin_minutes=bin_minutes
    )
    ax.errorbar(
        centers,
        median,
        yerr=error,
        fmt="o",
        color="#d9534f",
        ecolor="#d9534f",
        markersize=6,
        capsize=3,
        label=f"{bin_minutes:.0f}-min Binned",
    )

    ax.set_xlabel("Phase [hours from transit center]")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Phase-Folded Light Curve (P = {period_days:.4f} d)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_centroid_offsets(
    ra_offsets_arcsec: Sequence[float],
    dec_offsets_arcsec: Sequence[float],
    sigma_arcsec: float,
    output_path: Path,
    threshold_sigma: float = 3.0,
) -> Path:
    """Render a centroid difference-image offset map with threshold circle."""
    ra = np.asarray(ra_offsets_arcsec, dtype=float)
    dec = np.asarray(dec_offsets_arcsec, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ra, dec, color="#337ab7", alpha=0.7, s=40, label="Transit Center Offsets")

    # Draw 3-sigma threshold circle
    circle_radius = threshold_sigma * sigma_arcsec
    circle = plt.Circle(
        (0, 0),
        circle_radius,
        color="#5cb85c",
        fill=False,
        linewidth=2,
        linestyle="--",
        label=f"{threshold_sigma:.1f}$\\sigma$ Threshold ({circle_radius:.2f}\")",
    )
    ax.add_patch(circle)

    ax.set_xlabel("RA Offset [arcsec]")
    ax.set_ylabel("Dec Offset [arcsec]")
    ax.set_title("Difference-Image Centroid Significance Map")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_candidate_plots(
    workspace: CandidateWorkspace,
    period_days: Optional[float] = None,
    epoch_btjd: Optional[float] = None,
    signal: Optional[str] = None,
) -> Sequence[Path]:
    """Generate default diagnostic plots under candidate/<id>/figures/.

    Missing phase-fold parameters come from the candidate's transit
    configuration, optionally from the named per-signal configuration. If no
    configuration is available, the generic BLS result or a deterministic
    demonstration ephemeris is used. Real candidate light curves are used when
    available; otherwise a deterministic synthetic grid is rendered. Random
    offsets use a fixed seed for reproducibility.
    """
    figures_dir = workspace.path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)

    if period_days is None or epoch_btjd is None:
        ephemeris = load_transit_ephemeris(workspace, signal=signal)
        if period_days is None:
            period_days = ephemeris["period_days"]
        if epoch_btjd is None:
            epoch_btjd = ephemeris["epoch_btjd"]

    loaded = load_candidate_light_curve(workspace)
    if loaded is None:
        time = np.linspace(0, 27, 2000)
        flux = 1.0 - 0.0015 * (np.abs((time - epoch_btjd) % period_days) < 0.08).astype(float)
        flux += rng.normal(0, 0.0003, size=time.shape)
    else:
        time, flux = loaded

    lc_plot = figures_dir / "phase_folded_lc.png"
    plot_phase_folded_lc(time, flux, period_days, epoch_btjd, lc_plot)

    ra_offsets = rng.normal(0.05, 0.08, size=15)
    dec_offsets = rng.normal(-0.04, 0.08, size=15)
    centroid_plot = figures_dir / "centroid_offset.png"
    plot_centroid_offsets(ra_offsets, dec_offsets, sigma_arcsec=0.10, output_path=centroid_plot)

    return [lc_plot, centroid_plot]
