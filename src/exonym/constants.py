"""Canonical physical conversion constants used by EXONYM calculations.

The constants in this module are target-neutral conversion factors, not
measurements of a particular target. Keeping them in one import-only module
prevents small, inconsistent radius, density, and gravitational-constant
choices from silently changing otherwise identical calculations.

References
----------
* IAU Resolution B3 (2015), nominal solar and planetary conversion constants.
* CODATA 2018, Newtonian gravitational and proton constants.
* IAU Resolution B2 (2012), exact astronomical-unit definition.
"""

from __future__ import annotations

import math


# Time conversions are exact SI/JD conventions.
SECONDS_PER_DAY = 86_400.0
JULIAN_YEAR_DAYS = 365.25

# CODATA 2018. CGS follows from 1 m^3 kg^-1 = 10^3 cm^3 g^-1.
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11  # m^3 kg^-1 s^-2
GRAVITATIONAL_CONSTANT_CGS = GRAVITATIONAL_CONSTANT_SI * 1.0e3  # cm^3 g^-1 s^-2
BOLTZMANN_CONSTANT_J_K = 1.380649e-23  # exact, J K^-1
PROTON_MASS_KG = 1.67262192369e-27  # kg

# IAU 2015 Resolution B3 nominal conversion constants (exact by definition).
NOMINAL_SOLAR_RADIUS_M = 6.957e8
NOMINAL_SOLAR_RADIUS_KM = NOMINAL_SOLAR_RADIUS_M / 1.0e3
NOMINAL_SOLAR_MASS_PARAMETER_M3_S2 = 1.3271244e20
NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K = 5772.0
NOMINAL_EARTH_EQUATORIAL_RADIUS_M = 6.3781e6
NOMINAL_EARTH_EQUATORIAL_RADIUS_KM = NOMINAL_EARTH_EQUATORIAL_RADIUS_M / 1.0e3
NOMINAL_EARTH_MASS_PARAMETER_M3_S2 = 3.986004e14

# Derived conversion factors. The solar density is a reproducible
# computational normalization based on the nominal solar GM/radius and the
# declared CODATA G, rather than a separately rounded legacy literal.
SOLAR_MEAN_DENSITY_G_CM3 = (
    (NOMINAL_SOLAR_MASS_PARAMETER_M3_S2 / GRAVITATIONAL_CONSTANT_SI)
    / ((4.0 / 3.0) * math.pi * NOMINAL_SOLAR_RADIUS_M**3)
    / 1.0e3
)
EARTH_TO_SOLAR_RADIUS_RATIO = (
    NOMINAL_EARTH_EQUATORIAL_RADIUS_M / NOMINAL_SOLAR_RADIUS_M
)
EARTH_TO_SOLAR_MASS_PARAMETER_RATIO = (
    NOMINAL_EARTH_MASS_PARAMETER_M3_S2 / NOMINAL_SOLAR_MASS_PARAMETER_M3_S2
)
NOMINAL_SOLAR_LOGG_CGS = math.log10(
    (NOMINAL_SOLAR_MASS_PARAMETER_M3_S2 / NOMINAL_SOLAR_RADIUS_M**2) * 100.0
)

# Conventional Earth-mass/one-Julian-year Doppler planning normalization.
EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S = 0.0895

# IAU 2012 astronomical unit, converted through 1 pc = 648000/pi au.
ASTRONOMICAL_UNIT_M = 149_597_870_700.0
PARSEC_M = ASTRONOMICAL_UNIT_M * 648_000.0 / math.pi
