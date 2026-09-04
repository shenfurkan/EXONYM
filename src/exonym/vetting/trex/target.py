"""TargetScene -- stellar population context for TREX vetting.

Manages aperture stars, resolved neighbour catalogs, TRILEGAL background
star simulations, and contrast-curve data for a single TESS target.

The class is candidate-neutral: all target-specific data is passed via
constructor arguments or loaded from candidate-owned files.

Units, provenance, and fail-closed contract
-------------------------------------------
The scene contract implements the input side of Giacalone et al. (2021), ADS
``2021AJ....161...24G``, DOI ``10.3847/1538-3881/abd184``, with the background
population from Girardi et al. (2005), ADS ``2005A&A...436..895G``, DOI
``10.1051/0004-6361:20042352``.  Coordinates are ICRS degrees; mass/radius are
``M_sun``/``R_sun``; temperature is K; TESS magnitude and contrast are mag and
delta-mag; parallax is mas; contrast separation is arcsec; and background count
is an integer count.  The retained TRILEGAL cache and SHA-256 binding are
required.  Missing/nonfinite scene fields, nonmonotonic contrast curves,
duplicate neighbors, or a hash mismatch raises instead of supplying a fallback.
The scene is conditional scenario input, not source attribution or a claim.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError("{0} must be a finite number".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{0} must be a finite number".format(name)) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError("{0} must be a {1} number".format(name, qualifier))
    return number


class TargetScene:
    """Stellar population context for one TESS target.

    Attributes:
        tic_id: TIC identifier.
        ra_deg, dec_deg: Target coordinates [degrees].
        M_s_Msun: Target mass [Msun].
        R_s_Rsun: Target radius [Rsun].
        Teff_K: Target effective temperature [K].
        Tmag: TESS magnitude.
        plx_mas: Gaia parallax [mas].
        sectors: Observed TESS sectors.
        contrast_separations: [arcsec] from contrast curve.
        contrast_values: [delta_mag] from contrast curve.
        resolved_neighbors: List of dicts with M_s, R_s, delta_mag, separation.
        N_background: Number of background stars from TRILEGAL.
        trilegal_cache: Path to cached TRILEGAL CSV.
        background_sha256: Digest binding the retained background population.
    """

    def __init__(
        self,
        tic_id: int,
        ra_deg: float,
        dec_deg: float,
        M_s_Msun: float,
        R_s_Rsun: float,
        Teff_K: float,
        Tmag: float,
        plx_mas: float,
        sectors: List[int],
        contrast_separations: np.ndarray,
        contrast_values: np.ndarray,
        resolved_neighbors: List[Dict[str, float]],
        N_background: int,
        trilegal_cache: Path,
        background_sha256: str,
    ) -> None:
        if isinstance(tic_id, bool) or not isinstance(tic_id, int) or tic_id <= 0:
            raise ValueError("tic_id must be a positive integer")
        self.tic_id = tic_id
        self.ra_deg = _finite_number(ra_deg, "ra_deg")
        self.dec_deg = _finite_number(dec_deg, "dec_deg")
        if not 0.0 <= self.ra_deg < 360.0:
            raise ValueError("ra_deg must be in [0, 360)")
        if not -90.0 <= self.dec_deg <= 90.0:
            raise ValueError("dec_deg must be in [-90, 90]")
        self.M_s_Msun = _finite_number(M_s_Msun, "M_s_Msun", positive=True)
        self.R_s_Rsun = _finite_number(R_s_Rsun, "R_s_Rsun", positive=True)
        self.Teff_K = _finite_number(Teff_K, "Teff_K", positive=True)
        self.Tmag = _finite_number(Tmag, "Tmag")
        self.plx_mas = _finite_number(plx_mas, "plx_mas", positive=True)

        if not isinstance(sectors, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sectors
        ):
            raise ValueError("sectors must be a list of positive integers")
        self.sectors = list(sectors)

        self.contrast_separations = np.asarray(contrast_separations, dtype=float)
        self.contrast_values = np.asarray(contrast_values, dtype=float)
        if (
            self.contrast_separations.ndim != 1
            or self.contrast_values.ndim != 1
            or self.contrast_separations.size < 2
            or self.contrast_separations.shape != self.contrast_values.shape
            or not np.all(np.isfinite(self.contrast_separations))
            or not np.all(np.isfinite(self.contrast_values))
            or np.any(self.contrast_separations <= 0.0)
            or np.any(np.diff(self.contrast_separations) <= 0.0)
        ):
            raise ValueError(
                "contrast curves require at least two finite, increasing positive separations"
            )

        if not isinstance(resolved_neighbors, list):
            raise ValueError("resolved_neighbors must be a list")
        self.resolved_neighbors = []
        source_ids = set()
        for index, neighbor in enumerate(resolved_neighbors):
            if not isinstance(neighbor, dict):
                raise ValueError("resolved neighbor {0} must be an object".format(index))
            source_id = neighbor.get("source_id")
            if not isinstance(source_id, str) or not source_id or source_id in source_ids:
                raise ValueError("resolved neighbors must have unique source_id values")
            source_ids.add(source_id)
            self.resolved_neighbors.append(
                {
                    "source_id": source_id,
                    "M_s": _finite_number(neighbor.get("M_s"), "neighbor M_s", positive=True),
                    "R_s": _finite_number(neighbor.get("R_s"), "neighbor R_s", positive=True),
                    "delta_mag": _finite_number(neighbor.get("delta_mag"), "neighbor delta_mag"),
                    "separation_arcsec": _finite_number(
                        neighbor.get("separation_arcsec"),
                        "neighbor separation_arcsec",
                        positive=True,
                    ),
                }
            )

        if isinstance(N_background, bool) or not isinstance(N_background, int) or N_background < 0:
            raise ValueError("N_background must be a non-negative integer")
        self.N_background = N_background
        self.trilegal_cache = Path(trilegal_cache)
        if self.trilegal_cache.is_symlink() or not self.trilegal_cache.is_file():
            raise ValueError("trilegal_cache must be an available regular file")
        if not isinstance(background_sha256, str) or len(background_sha256) != 64:
            raise ValueError("background_sha256 must be a SHA-256 digest")
        if _sha256(self.trilegal_cache) != background_sha256:
            raise ValueError("trilegal_cache does not match background_sha256")
        self.background_sha256 = background_sha256

    @property
    def n_neighbors(self) -> int:
        return len(self.resolved_neighbors)

    @property
    def has_contrast_data(self) -> bool:
        return True

    def verify_background(self) -> None:
        """Reject a background population that changed after scene construction."""
        if self.trilegal_cache.is_symlink() or not self.trilegal_cache.is_file():
            raise ValueError("trilegal_cache is no longer an available regular file")
        if _sha256(self.trilegal_cache) != self.background_sha256:
            raise ValueError("trilegal_cache changed after scene construction")

    def neighbor_masses_radii(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (masses, radii, delta_mags) for resolved neighbors.

        All neighbour properties were explicitly supplied in the scene manifest.
        """
        n = len(self.resolved_neighbors)
        masses = np.full(n, np.nan)
        radii = np.full(n, np.nan)
        delta_mags = np.full(n, np.nan)

        for i, nb in enumerate(self.resolved_neighbors):
            delta_mags[i] = float(nb["delta_mag"])
            masses[i] = float(nb["M_s"])
            radii[i] = float(nb["R_s"])

        return masses, radii, delta_mags

    def companion_delta_mags(self) -> np.ndarray:
        """Delta magnitudes of all resolved neighbours."""
        if not self.resolved_neighbors:
            return np.array([])
        return np.array(
            [float(nb["delta_mag"]) for nb in self.resolved_neighbors],
            dtype=float,
        )

    def neighbor_dicts_for_evidence(self) -> List[Dict[str, float]]:
        """Return list of {M_s, R_s} dicts for evidence calculation."""
        masses, radii, _ = self.neighbor_masses_radii()
        return [
            {"M_s": float(m), "R_s": float(r)}
            for m, r in zip(masses, radii)
            if np.isfinite(m) and np.isfinite(r)
        ]


__all__ = ["TargetScene"]
