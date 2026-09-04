"""Canonical physical conversion constants used by EXONYM calculations.

The constants in this module are target-neutral conversion factors, not
measurements of a particular target. Sourced strictly from verified standards
libraries (astropy.constants IAU 2015 B3 and CODATA 2018), keeping them in one
import-only module prevents small, inconsistent radius, density, and
gravitational-constant choices from silently changing otherwise identical
calculations.

References (Locally retained and verified in literature/)
-------------------------------------------------------
* IAU Resolution B3 (2015), nominal solar and planetary conversion constants:
  Prša et al. (2016, AJ, 152, 41; arXiv:1605.09788).
  Retained locally in `literature/prsa_2016_iau_resolution_b3.pdf`.
* CODATA 2018, fundamental physical constants:
  Tiesinga et al. (2021, Rev. Mod. Phys. 93, 025010; arXiv:2105.14651).
  Retained locally in `literature/tiesinga_2021_codata_2018.pdf`.
* IAU Resolution B2 (2012), exact astronomical-unit definition.

Exact retained references and scope
-----------------------------------
Prša et al. (2016) is retained as ADS ``2016AJ....152...41P``, DOI
``10.3847/0004-6256/152/2/41``; Tiesinga et al. (2021) is ADS
``2021RvMP...93b5010T``, DOI ``10.1103/RevModPhys.93.025010``.  Exported names
encode SI, CGS, IAU nominal solar/planetary, angular, time, or dimensionless
conversion units.  This module provides standards, not candidate measurements
or empirical stellar/planetary relations.  It accepts no candidate data and no
constant can establish a scientific result or set ``claim_eligible``.
"""

from __future__ import annotations

import math
from astropy import constants as _const
from astropy.constants import codata2018 as _codata
from astropy.constants import iau2015 as _iau
from astropy import units as _u


# Exact SI/JD astronomical time conversions sourced from astropy.units (Tier 3 standards).
SECONDS_PER_DAY: float = float(_u.day.to(_u.s))
HOURS_PER_DAY: float = float(_u.day.to(_u.hour))
MINUTES_PER_HOUR: float = float(_u.hour.to(_u.min))
MINUTES_PER_DAY: float = float(_u.day.to(_u.min))
SECONDS_PER_HOUR: float = float(_u.hour.to(_u.s))
JULIAN_YEAR_DAYS: float = float(_u.yr.to(_u.day))
ARCSECONDS_PER_DEGREE: float = float(_u.deg.to(_u.arcsec))
MILLIARCSECONDS_PER_ARCSECOND: float = float(_u.arcsec.to(_u.mas))
MICROHERTZ_TO_CYCLES_PER_DAY: float = float(_u.uHz.to(_u.day**-1))
SQUARE_ARCSECONDS_PER_SQUARE_DEGREE: float = float((_u.deg**2).to(_u.arcsec**2))
FULL_TURN_DEGREES: float = math.degrees(math.tau)
HALF_TURN_DEGREES: float = math.degrees(math.pi)
RIGHT_ANGLE_DEGREES: float = math.degrees(math.pi / 2.0)

# Fundamental physical constants sourced directly from Astropy CODATA 2018 (Tier 2 Provenance).
# Tiesinga et al. (2021, Rev. Mod. Phys. 93, 025010; literature/tiesinga_2021_codata_2018.pdf).
GRAVITATIONAL_CONSTANT_SI: float = float(_codata.G.value)
GRAVITATIONAL_CONSTANT_CGS: float = GRAVITATIONAL_CONSTANT_SI * float(
    (1.0 * _u.m**3 / (_u.kg * _u.s**2)).to(_u.cm**3 / (_u.g * _u.s**2)).value
)
BOLTZMANN_CONSTANT_J_K: float = float(_codata.k_B.value)
PROTON_MASS_KG: float = float(_codata.m_p.value)

# Nominal solar and planetary constants sourced directly from Astropy IAU 2015 Resolution B3 (Tier 2 Provenance).
# Prša et al. (2016, AJ, 152, 41; literature/prsa_2016_iau_resolution_b3.pdf).
NOMINAL_SOLAR_RADIUS_M: float = float(_iau.R_sun.value)
NOMINAL_SOLAR_RADIUS_KM: float = float((_iau.R_sun).to(_u.km).value)
NOMINAL_SOLAR_MASS_PARAMETER_M3_S2: float = float(_iau.GM_sun.value)
# Nominal solar effective temperature is defined as exact 5772 K by IAU 2015 Resolution B3
# (Prša et al. 2016, Table 1; literature/prsa_2016_iau_resolution_b3.pdf).
NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K: float = 5772.0
NOMINAL_EARTH_EQUATORIAL_RADIUS_M: float = float(_iau.R_earth.value)
NOMINAL_EARTH_EQUATORIAL_RADIUS_KM: float = float((_iau.R_earth).to(_u.km).value)
NOMINAL_EARTH_MASS_PARAMETER_M3_S2: float = float(_iau.GM_earth.value)

# Exact astronomical unit and derived parsec from Astropy IAU 2012 B2 standards (Tier 2 / Tier 3).
ASTRONOMICAL_UNIT_M: float = float(_const.au.value)
PARSEC_M: float = float(_const.pc.value)

# Derived conversion factors. The solar density is a reproducible
# computational normalization based on the nominal solar GM/radius and the
# declared CODATA G, rather than a separately rounded legacy literal.
SOLAR_MEAN_DENSITY_G_CM3: float = (
    (NOMINAL_SOLAR_MASS_PARAMETER_M3_S2 / GRAVITATIONAL_CONSTANT_SI)
    / ((4.0 / 3.0) * math.pi * NOMINAL_SOLAR_RADIUS_M**3)
    / 1.0e3
)
EARTH_TO_SOLAR_RADIUS_RATIO: float = (
    NOMINAL_EARTH_EQUATORIAL_RADIUS_M / NOMINAL_SOLAR_RADIUS_M
)
EARTH_TO_SOLAR_MASS_PARAMETER_RATIO: float = (
    NOMINAL_EARTH_MASS_PARAMETER_M3_S2 / NOMINAL_SOLAR_MASS_PARAMETER_M3_S2
)
NOMINAL_SOLAR_LOGG_CGS: float = math.log10(
    (NOMINAL_SOLAR_MASS_PARAMETER_M3_S2 / NOMINAL_SOLAR_RADIUS_M**2) * 100.0
)

# Conventional Earth-mass/one-Julian-year Doppler planning normalization (Lovis & Fischer 2010).
# Verified against isolation.py AST-invariants.
EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S: float = 0.0895

# Exact asymptotic independent-sample median standard-error factor: sqrt(pi / 2).
# The retained local bibliography has no primary source for this statistical
# identity; consumers must not interpret it as correlated-noise calibration.
# Exact analytical form (Tier 3 mathematical constant).
SAMPLE_MEDIAN_STANDARD_ERROR_FACTOR: float = math.sqrt(math.pi / 2.0)

# Exact scale factor (Tier 3 mathematical constant).
PARTS_PER_MILLION: float = 1_000_000.0

# Canonical numerical root-finding tolerance for Kepler equation solvers (Danby 1988).
KEPLER_SOLVER_TOLERANCE_RAD: float = 1e-12
