import math

import pytest

from exonym.constants import (
    EARTH_TO_SOLAR_MASS_PARAMETER_RATIO,
    EARTH_TO_SOLAR_RADIUS_RATIO,
    GRAVITATIONAL_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_SI,
    NOMINAL_EARTH_EQUATORIAL_RADIUS_M,
    NOMINAL_SOLAR_RADIUS_M,
    PARSEC_M,
    SOLAR_MEAN_DENSITY_G_CM3,
)


def test_canonical_conversion_constants_are_self_consistent():
    assert GRAVITATIONAL_CONSTANT_CGS == pytest.approx(GRAVITATIONAL_CONSTANT_SI * 1.0e3)
    assert EARTH_TO_SOLAR_RADIUS_RATIO == pytest.approx(
        NOMINAL_EARTH_EQUATORIAL_RADIUS_M / NOMINAL_SOLAR_RADIUS_M
    )
    assert EARTH_TO_SOLAR_MASS_PARAMETER_RATIO == pytest.approx(3.00348934885e-6, rel=1e-11)
    assert SOLAR_MEAN_DENSITY_G_CM3 == pytest.approx(1.4097798243, rel=1e-10)
    assert PARSEC_M == pytest.approx(3.085677581491367e16, rel=1e-15)
    assert math.isfinite(PARSEC_M)
