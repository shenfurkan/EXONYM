"""Cross-platform physical constants for the TREX statistical vetting engine.

Constants are drawn from IAU 2015 Resolution B3 nominal values and
CODATA 2018 through the official ``astropy.constants`` / ``astropy.units``
interfaces; every exported value is the reference float64 quantity, never a
hand-written or truncated literal (AGENTS.md Rules 5, 10 & 13).  CGS units
are used throughout to match the astrophysical literature conventions of
Giacalone et al. (2021).

References
----------
* IAU Resolution B3 (2015): nominal solar and planetary conversion constants.
* CODATA 2018: Newtonian gravitational constant and related values.
* IAU 2012 Resolution B2: astronomical unit (149 597 870 700 m, exact).
"""

from __future__ import annotations

import math as _math

import astropy.units as _u
from astropy import constants as _const
from astropy.constants import codata2018 as _codata
from astropy.constants import iau2015 as _iau

# ---- SI base constants (CODATA 2018) ----
GRAVITATIONAL_CONSTANT_SI: float = float(_codata.G.value)          # m^3 kg^-1 s^-2

# ---- CGS constants (official astropy conversions) ----
GRAVITATIONAL_CONSTANT_CGS: float = float(_codata.G.cgs.value)     # cm^3 g^-1 s^-2
SPEED_OF_LIGHT_CGS: float = float(_const.c.cgs.value)              # cm s^-1 (exact)

# ---- IAU 2015 Resolution B3 nominal values ----
NOMINAL_SOLAR_RADIUS_CM: float = float(_iau.R_sun.cgs.value)       # cm
NOMINAL_SOLAR_RADIUS_M: float = float(_iau.R_sun.value)            # m
NOMINAL_SOLAR_MASS_G: float = float((_iau.GM_sun / _codata.G).cgs.value)  # g (exact GM / G)
NOMINAL_SOLAR_MASS_PARAMETER_CGS: float = float(_iau.GM_sun.cgs.value)    # cm^3 s^-2
NOMINAL_EARTH_EQUATORIAL_RADIUS_CM: float = float(_iau.R_earth.cgs.value)  # cm
NOMINAL_EARTH_EQUATORIAL_RADIUS_KM: float = float(_iau.R_earth.to_value(_u.km))
NOMINAL_EARTH_MASS_PARAMETER_CGS: float = float(_iau.GM_earth.cgs.value)  # cm^3 s^-2

# ---- Derived conversion ratios ----
EARTH_TO_SOLAR_RADIUS_RATIO: float = (
    NOMINAL_EARTH_EQUATORIAL_RADIUS_CM / NOMINAL_SOLAR_RADIUS_CM
)

# ---- Astronomical unit (IAU 2012 Resolution B2) ----
ASTRONOMICAL_UNIT_CM: float = float(_const.au.cgs.value)          # cm

# ---- Time ----
SECONDS_PER_DAY: float = float(_u.day.to(_u.s))

# ---- Solar mean density (CGS) ----
SOLAR_MEAN_DENSITY_G_CM3: float = (
    (NOMINAL_SOLAR_MASS_PARAMETER_CGS / GRAVITATIONAL_CONSTANT_CGS)
    / ((4.0 / 3.0) * _math.pi * NOMINAL_SOLAR_RADIUS_CM ** 3)
)

# ---- Convenient short-names matching the TRICERATOPS literature convention ----
Msun: float = NOMINAL_SOLAR_MASS_G
Rsun: float = NOMINAL_SOLAR_RADIUS_CM
Rearth: float = NOMINAL_EARTH_EQUATORIAL_RADIUS_CM
G: float = GRAVITATIONAL_CONSTANT_CGS
au: float = ASTRONOMICAL_UNIT_CM
pi: float = _math.pi


__all__ = [
    "GRAVITATIONAL_CONSTANT_CGS",
    "SPEED_OF_LIGHT_CGS",
    "NOMINAL_SOLAR_RADIUS_CM",
    "NOMINAL_SOLAR_MASS_G",
    "NOMINAL_SOLAR_MASS_PARAMETER_CGS",
    "NOMINAL_EARTH_EQUATORIAL_RADIUS_CM",
    "EARTH_TO_SOLAR_RADIUS_RATIO",
    "ASTRONOMICAL_UNIT_CM",
    "SECONDS_PER_DAY",
    "SOLAR_MEAN_DENSITY_G_CM3",
    "Msun",
    "Rsun",
    "Rearth",
    "G",
    "au",
    "pi",
]