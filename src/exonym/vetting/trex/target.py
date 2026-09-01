"""TargetScene -- stellar population context for TREX vetting.

Manages aperture stars, resolved neighbour catalogs, TRILEGAL background
star simulations, and contrast-curve data for a single TESS target.

The class is candidate-neutral: all target-specific data is passed via
constructor arguments or loaded from candidate-owned files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import Msun, Rsun, pi, G, au
from .funcs import stellar_relations, delta_mag_to_flux_ratio


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
    """

    def __init__(
        self,
        tic_id: int,
        ra_deg: float,
        dec_deg: float,
        M_s_Msun: float = 1.0,
        R_s_Rsun: float = 1.0,
        Teff_K: float = 5772.0,
        Tmag: float = 10.0,
        plx_mas: float = 1.0,
        sectors: Optional[List[int]] = None,
        contrast_separations: Optional[np.ndarray] = None,
        contrast_values: Optional[np.ndarray] = None,
        resolved_neighbors: Optional[List[Dict[str, float]]] = None,
        N_background: int = 0,
        trilegal_cache: Optional[Path] = None,
    ) -> None:
        self.tic_id = tic_id
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.M_s_Msun = M_s_Msun
        self.R_s_Rsun = R_s_Rsun
        self.Teff_K = Teff_K
        self.Tmag = Tmag
        self.plx_mas = plx_mas
        self.sectors = sectors or []
        self.contrast_separations = (
            np.asarray(contrast_separations, dtype=float)
            if contrast_separations is not None
            else None
        )
        self.contrast_values = (
            np.asarray(contrast_values, dtype=float)
            if contrast_values is not None
            else None
        )
        self.resolved_neighbors = resolved_neighbors or []
        self.N_background = N_background
        self.trilegal_cache = trilegal_cache

    @property
    def n_neighbors(self) -> int:
        return len(self.resolved_neighbors)

    @property
    def has_contrast_data(self) -> bool:
        return (
            self.contrast_separations is not None
            and self.contrast_values is not None
            and len(self.contrast_separations) > 1
        )

    def neighbor_masses_radii(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (masses, radii, delta_mags) for resolved neighbors.

        If neighbours have no mass/radius, estimate from delta_mag.
        """
        n = len(self.resolved_neighbors)
        masses = np.full(n, np.nan)
        radii = np.full(n, np.nan)
        delta_mags = np.full(n, np.nan)

        for i, nb in enumerate(self.resolved_neighbors):
            delta_mags[i] = float(nb.get("delta_mag", 0.0))
            if "M_s" in nb and "R_s" in nb:
                masses[i] = float(nb["M_s"])
                radii[i] = float(nb["R_s"])
            else:
                # Crude estimate from delta_mag and main-sequence relation
                dm = delta_mags[i]
                est_mass = max(0.1, self.M_s_Msun * 10 ** (-0.2 * dm))
                masses[i] = est_mass
                r_est, _ = stellar_relations(np.array([est_mass]))
                radii[i] = float(r_est[0])

        return masses, radii, delta_mags

    def companion_delta_mags(self) -> np.ndarray:
        """Delta magnitudes of all resolved neighbours."""
        if not self.resolved_neighbors:
            return np.array([])
        return np.array(
            [float(nb.get("delta_mag", 0.0)) for nb in self.resolved_neighbors],
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