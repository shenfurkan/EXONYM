"""Headless diagnostic vetting figure generation.

All routines enforce headless rendering (`matplotlib.use('Agg')`) to run cleanly
in automated pipelines and CI/CD without requiring display servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Enforce non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from .inputs import load_transit_ephemeris
from .lightcurve import bin_phase_folded_flux, phase_hours
from .search import load_candidate_light_curve
from .workspace import CandidateWorkspace, validate_signal_suffix


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
    """Render supplied centroid-offset samples and an uncertainty threshold."""
    ra = np.asarray(ra_offsets_arcsec, dtype=float)
    dec = np.asarray(dec_offsets_arcsec, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ra, dec, color="#337ab7", alpha=0.7, s=40, label="Supplied offset samples")

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
    ax.set_title("Centroid offset samples")
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


def plot_mcmc_corner(
    chain: np.ndarray,
    output_path: Path,
    labels: Optional[Sequence[str]] = None,
) -> Path:
    """Render an MCMC posterior corner plot using corner.py."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import corner
    except ImportError as exc:
        raise RuntimeError("corner is not installed. Install with: pip install corner") from exc

    samples = np.asarray(chain, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 5:
        raise ValueError("samples chain must be a 2D array with at least 5 samples")

    if labels is None:
        if samples.shape[1] == 7:
            labels = [
                "$R_p/R_\\star$",
                "$\\log_{10}\\rho_\\star$",
                "$b$",
                "Flux Base",
                "$\\log\\sigma_j$",
                "$q_1$",
                "$q_2$",
            ]
        elif samples.shape[1] == 9:
            labels = [
                "$R_p/R_\\star$",
                "$\\log_{10}\\rho_\\star$",
                "$b$",
                "Flux Base",
                "$\\log\\sigma_j$",
                "$q_1$",
                "$q_2$",
                "$\\sqrt{e}\\cos\\omega$",
                "$\\sqrt{e}\\sin\\omega$",
            ]
        else:
            labels = [f"Param {i+1}" for i in range(samples.shape[1])]

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_candidate_plots(
    workspace: CandidateWorkspace,
    period_days: Optional[float] = None,
    epoch_btjd: Optional[float] = None,
    signal: Optional[str] = None,
    include_corner: bool = False,
) -> Sequence[Path]:
    """Generate default diagnostic plots under candidate/<id>/figures/.

    Missing phase-fold parameters come from the candidate's transit
    configuration, optionally from the named per-signal configuration. The
    command requires a candidate-data ephemeris and light curve. Corner plots
    require the matching saved MCMC chain.
    """
    signal = validate_signal_suffix(signal)
    figures_dir = workspace.path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if period_days is None or epoch_btjd is None:
        ephemeris = load_transit_ephemeris(workspace, signal=signal)
        if ephemeris.get("source") == "synthetic-demo":
            raise ValueError("candidate plot requires a candidate-data ephemeris")
        if period_days is None:
            period_days = ephemeris["period_days"]
        if epoch_btjd is None:
            epoch_btjd = ephemeris["epoch_btjd"]

    loaded = load_candidate_light_curve(workspace)
    if loaded is None:
        raise ValueError("candidate plot requires a readable candidate light curve")
    time, flux = loaded

    suffix = f".{signal.lstrip('.')}" if signal else ""
    lc_plot = figures_dir / f"phase_folded_lc{suffix}.png"
    plot_phase_folded_lc(time, flux, period_days, epoch_btjd, lc_plot)

    results: List[Path] = [lc_plot]

    if include_corner:
        chain_path = workspace.path / "outputs" / f"mcmc_transit_fit_chain{suffix}.npy"
        corner_plot = figures_dir / f"corner_plot{suffix}.png"
        if not chain_path.is_file():
            raise ValueError(
                "corner plot requires an existing MCMC fit chain: {0}".format(
                    chain_path.relative_to(workspace.path)
                )
            )
        chain = np.load(str(chain_path))
        plot_mcmc_corner(chain, output_path=corner_plot)
        results.append(corner_plot)

    return results
