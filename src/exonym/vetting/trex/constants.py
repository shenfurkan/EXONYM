"""Cross-platform physical constants for the TREX statistical vetting engine.

Constants are drawn from IAU 2015 Resolution B3 nominal values and
CODATA 2018.  CGS units are used throughout to match the astrophysical
literature conventions of Giacalone et al. (2021).

References
----------
* IAU Resolution B3 (2015): nominal solar and planetary conversion constants.
* CODATA 2018: Newtonian gravitational constant and related values.
"""

from __future__ import annotations

import math as _math

# ---- SI base constants (CODATA 2018) ----
GRAVITATIONAL_CONSTANT_SI: float = 6.67430e-11          # m^3 kg^-1 s^-2

# ---- CGS constants (derived from SI) ----
GRAVITATIONAL_CONSTANT_CGS: float = GRAVITATIONAL_CONSTANT_SI * 1.0e3  # cm^3 g^-1 s^-2
SPEED_OF_LIGHT_CGS: float = 2.99792458e10                # cm s^-1 (exact)

# ---- IAU 2015 Resolution B3 nominal values ----
NOMINAL_SOLAR_RADIUS_CM: float = 6.957e10                # cm
NOMINAL_SOLAR_RADIUS_M: float = 6.957e8                  # m
NOMINAL_SOLAR_MASS_G: float = 1.9884e33                  # g  (derived from GM / G)
NOMINAL_SOLAR_MASS_PARAMETER_CGS: float = 1.3271244e26   # cm^3 s^-2 (GM_sun)
NOMINAL_EARTH_EQUATORIAL_RADIUS_CM: float = 6.3781e8     # cm
NOMINAL_EARTH_EQUATORIAL_RADIUS_KM: float = 6.3781e3     # km
NOMINAL_EARTH_MASS_PARAMETER_CGS: float = 3.986004e20    # cm^3 s^-2 (GM_earth)

# ---- Derived conversion ratios ----
EARTH_TO_SOLAR_RADIUS_RATIO: float = (
    NOMINAL_EARTH_EQUATORIAL_RADIUS_CM / NOMINAL_SOLAR_RADIUS_CM
)

# ---- Astronomical unit ----
ASTRONOMICAL_UNIT_CM: float = 1.49597870700e13           # cm (IAU 2012 exact)

# ---- Time ----
SECONDS_PER_DAY: float = 86_400.0

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