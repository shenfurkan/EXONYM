"""Deterministic, target-neutral astrophysical gold-standard regressions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gold_standards"
_FIXTURE_REQUIRED_FIELDS = frozenset(("schema_version", "fixture_status", "reference", "provenance"))
_FIXTURE_PROVENANCE_FIELDS = frozenset(("source_kind", "independence", "limitations"))
_GOLD_STANDARD_FIXTURES = frozenset(
    {
        "claret_limb_darkening_sample.json",
        "kepler_rv_eccentric_benchmarks.json",
        "mandel_agol_quadrature_benchmarks.json",
        "mist_synthetic_stellar_nodes.json",
        "phasecurve_circular_harmonic_benchmarks.json",
        "seismic_scaling_standard_stars.json",
        "ttv_first_order_mmr_benchmarks.json",
    }
)
_TAU = 2.0 * math.pi
_SPEED_OF_LIGHT_M_S = 299_792_458.0
_PLANCK_CONSTANT_J_S = 6.62607015e-34
_BOLTZMANN_CONSTANT_J_K = 1.380649e-23


def _reject_duplicate_fixture_keys(pairs):
    """Reject ambiguous JSON objects before fixture values can be trusted."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def test_gold_standard_fixture_inventory_is_explicit_and_complete():
    assert {path.name for path in FIXTURE_ROOT.glob("*.json")} == _GOLD_STANDARD_FIXTURES


def _reject_nonfinite_fixture_constant(value):
    """Reject JSON extensions such as NaN and Infinity in reference data."""
    raise ValueError("non-finite JSON number: {0}".format(value))


def _parse_finite_fixture_float(value):
    """Parse a finite decimal fixture value without accepting overflow."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _assert_finite_fixture_value(value):
    """Reject non-finite integers or floats that passed JSON parsing."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except OverflowError as exc:
            raise ValueError("fixture contains an overflowing number") from exc
        if not finite:
            raise ValueError("fixture contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite_fixture_value(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_fixture_value(item)
        return
    raise ValueError("fixture contains an unsupported JSON value")


def _validate_fixture_metadata(payload, filename):
    """Require provenance fields that distinguish analytic and literature fixtures."""
    if not isinstance(payload, dict):
        raise ValueError("gold-standard fixture must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("gold-standard fixture must declare schema_version 1")
    missing = _FIXTURE_REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError("gold-standard fixture is missing required fields: {0}".format(", ".join(sorted(missing))))
    if not isinstance(payload["fixture_status"], str) or not payload["fixture_status"]:
        raise ValueError("gold-standard fixture must declare a non-empty fixture_status")
    if not isinstance(payload["reference"], str) or not payload["reference"]:
        raise ValueError("gold-standard fixture must declare a non-empty reference")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("gold-standard fixture provenance must be an object")
    missing_provenance = _FIXTURE_PROVENANCE_FIELDS.difference(provenance)
    if missing_provenance:
        raise ValueError(
            "gold-standard fixture provenance is missing fields: {0}".format(
                ", ".join(sorted(missing_provenance))
            )
        )
    if not all(isinstance(provenance[name], str) and provenance[name] for name in _FIXTURE_PROVENANCE_FIELDS):
        raise ValueError("gold-standard fixture provenance fields must be non-empty strings")
    _assert_finite_fixture_value(payload)
    return payload


def _load_fixture(filename: str):
    """Load strict, finite, provenance-declared JSON without network access."""
    if Path(filename).name != filename:
        raise ValueError("gold-standard fixture filename must not contain path components")
    path = FIXTURE_ROOT / filename
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_fixture_constant,
            parse_float=_parse_finite_fixture_float,
            object_pairs_hook=_reject_duplicate_fixture_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("gold-standard fixture is not strict finite JSON: {0}".format(path.name)) from exc
    return _validate_fixture_metadata(payload, filename)


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ('{"schema_version": 1, "schema_version": 1}', "strict finite JSON"),
        ('{"schema_version": NaN}', "strict finite JSON"),
        ('{"schema_version": 1}', "missing required fields"),
    ),
)
def test_gold_standard_fixture_loader_rejects_ambiguous_or_incomplete_json(
    tmp_path, monkeypatch, content, message
):
    """Reference fixtures must reject duplicate keys, non-finite values, and missing provenance."""
    filename = "claret_limb_darkening_sample.json"
    (tmp_path / filename).write_text(content, encoding="utf-8")
    monkeypatch.setitem(globals(), "FIXTURE_ROOT", tmp_path)

    with pytest.raises(ValueError, match=message):
        _load_fixture(filename)


def _covered_arc_radians(radius, separation, planet_radius):
    """Return the covered angular arc of one stellar ring."""
    if separation == 0.0:
        return 2.0 * math.pi if radius <= planet_radius else 0.0
    if separation + radius <= planet_radius:
        return 2.0 * math.pi
    if separation >= radius + planet_radius or radius >= separation + planet_radius:
        return 0.0
    cosine = (radius**2 + separation**2 - planet_radius**2) / (2.0 * radius * separation)
    return 2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0)))


def _covered_angular_integral(arc_radians, nodes, weights):
    """Integrate one analytically bounded occulted angular interval."""
    if arc_radians == 0.0:
        return 0.0
    angles = 0.5 * arc_radians * nodes
    integrand = np.ones_like(angles)
    return 0.5 * arc_radians * float(np.sum(weights * integrand))


def _instantaneous_transit_quadrature(case, phase_days, radial_nodes, angular_nodes_count):
    """Integrate a quadratic stellar disk over its analytically bounded occulted region."""
    period_days = float(case["period_days"])
    radius_ratio = float(case["rp_rs"])
    a_rs = float(case["a_rs"])
    impact_parameter = float(case["impact_parameter"])
    q1 = float(case["q1"])
    q2 = float(case["q2"])
    root_q1 = math.sqrt(q1)
    u1 = 2.0 * root_q1 * q2
    u2 = root_q1 * (1.0 - 2.0 * q2)
    inclination = math.acos(impact_parameter / a_rs)
    orbital_angle = _TAU * float(phase_days) / period_days
    projected_x = a_rs * math.sin(orbital_angle)
    projected_y = a_rs * math.cos(inclination) * math.cos(orbital_angle)
    separation = math.hypot(projected_x, projected_y)

    breaks = [0.0, 1.0]
    for boundary in (abs(separation - radius_ratio), separation + radius_ratio):
        if 0.0 < boundary < 1.0:
            breaks.append(boundary)
    breaks = sorted(set(breaks))
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes)
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(angular_nodes_count)
    blocked = 0.0
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        radii = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        mu = np.sqrt(1.0 - radii**2)
        intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        angular_integrals = np.asarray(
            [_covered_angular_integral(_covered_arc_radians(float(radius), separation, radius_ratio), angular_nodes, angular_weights) for radius in radii]
        )
        blocked += 0.5 * (upper - lower) * float(
            np.sum(weights * intensity * angular_integrals * radii)
        )

    total = math.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    return float(case["baseline"]) * (1.0 - blocked / total)


def _quadrature_transit_flux(case, quadrature):
    """Evaluate the independent disk quadrature at high-resolution finite exposure."""
    exposure_days = float(case["exposure_seconds"]) / 86400.0
    subsamples = int(quadrature["reference_exposure_subsamples"])
    radial_nodes = int(quadrature["radial_nodes"])
    angular_nodes = int(quadrature["angular_nodes"])
    offsets = ((np.arange(subsamples, dtype=float) + 0.5) / subsamples - 0.5) * exposure_days
    return np.asarray(
        [
            np.mean(
                [
                    _instantaneous_transit_quadrature(
                        case, phase + offset, radial_nodes, angular_nodes
                    )
                    for offset in offsets
                ]
            )
            for phase in case["phase_days"]
        ],
        dtype=float,
    )


def _independent_eccentric_anomaly_rad(mean_anomaly_rad, eccentricity, iterations=128):
    """Solve Kepler's monotonic elliptic equation by bisection, independently of Halley."""
    reduced_mean_anomaly = float(mean_anomaly_rad) % _TAU
    if reduced_mean_anomaly == 0.0:
        return 0.0
    lower, upper = 0.0, _TAU
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        residual = midpoint - eccentricity * math.sin(midpoint) - reduced_mean_anomaly
        if residual < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _independent_keplerian_velocity_m_per_s(case):
    """Evaluate a Keplerian RV curve with bisection and sine/cosine anomaly relations."""
    eccentricity = float(case["eccentricity"])
    times = np.asarray(case["time_bjd_tdb"], dtype=float)
    mean_anomaly = float(case["mean_anomaly_reference_rad"]) + _TAU * (
        times - float(case["reference_time_bjd_tdb"])
    ) / float(case["period_days"])
    eccentric_anomaly = np.asarray(
        [_independent_eccentric_anomaly_rad(value, eccentricity) for value in mean_anomaly],
        dtype=float,
    )
    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    true_anomaly = np.arctan2(
        math.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly) / denominator,
        (np.cos(eccentric_anomaly) - eccentricity) / denominator,
    )
    return float(case["semi_amplitude_m_per_s"]) * (
        np.cos(true_anomaly + float(case["argument_periastron_rad"]))
        + eccentricity * math.cos(float(case["argument_periastron_rad"]))
    )


def _independent_blackbody_magnitudes(node, bands, bandpasses, extinction_ratios):
    """Evaluate the fixture Planck law from exact SI constants, outside the production helper."""
    teff_k = float(node["teff_k"])
    radius_distance = math.exp(float(node["log_radius_over_distance"]))
    magnitudes = []
    for band in bands:
        passband = bandpasses[band]
        wavelength_m = float(passband["pivot_wavelength_micron"]) * 1.0e-6
        frequency_hz = _SPEED_OF_LIGHT_M_S / wavelength_m
        exponent = _PLANCK_CONSTANT_J_S * frequency_hz / (_BOLTZMANN_CONSTANT_J_K * teff_k)
        intensity_w_m2_hz_sr = (
            2.0 * _PLANCK_CONSTANT_J_S * frequency_hz**3 / _SPEED_OF_LIGHT_M_S**2 / math.expm1(exponent)
        )
        flux_jy = math.pi * intensity_w_m2_hz_sr * radius_distance**2 / 1.0e-26
        magnitudes.append(
            -2.5 * math.log10(flux_jy / float(passband["vega_zero_point_jy"]))
            + float(node["av_mag"]) * float(extinction_ratios[band])
        )
    return np.asarray(magnitudes, dtype=float)


def _true_anomaly_to_mean_anomaly(true_anomaly, eccentricity):
    """Convert true anomaly to a principal mean anomaly for an ellipse."""
    eccentric_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - eccentricity) * math.sin(true_anomaly / 2.0),
        math.sqrt(1.0 + eccentricity) * math.cos(true_anomaly / 2.0),
    )
    return eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)


def test_constants_and_iau_conversions_match_canonical_standards():
    """
    NEDEN: Farklı modüllerde farklı yuvarlanmış G veya R_sun sabitlerinin kullanılması,
    yoğunluk ve kütle hesaplarında küçük ama biriken sistematik hatalar yaratır.
    IAU 2015 B3 ve CODATA 2018 standartlarının analitik tutarlılığı doğrulanır.
    """
    from exonym.constants import (
        ASTRONOMICAL_UNIT_M,
        EARTH_TO_SOLAR_MASS_PARAMETER_RATIO,
        EARTH_TO_SOLAR_RADIUS_RATIO,
        GRAVITATIONAL_CONSTANT_CGS,
        GRAVITATIONAL_CONSTANT_SI,
        NOMINAL_EARTH_EQUATORIAL_RADIUS_M,
        NOMINAL_EARTH_MASS_PARAMETER_M3_S2,
        NOMINAL_SOLAR_MASS_PARAMETER_M3_S2,
        NOMINAL_SOLAR_RADIUS_M,
        PARSEC_M,
        SOLAR_MEAN_DENSITY_G_CM3,
    )

    assert GRAVITATIONAL_CONSTANT_SI == pytest.approx(6.67430e-11, rel=0.0, abs=1e-25)
    assert GRAVITATIONAL_CONSTANT_CGS == pytest.approx(6.67430e-8, rel=0.0, abs=1e-22)
    assert NOMINAL_SOLAR_RADIUS_M == pytest.approx(6.957e8, rel=0.0, abs=0.0)
    assert ASTRONOMICAL_UNIT_M == pytest.approx(149597870700.0, rel=0.0, abs=0.0)
    assert EARTH_TO_SOLAR_RADIUS_RATIO == pytest.approx(
        NOMINAL_EARTH_EQUATORIAL_RADIUS_M / NOMINAL_SOLAR_RADIUS_M,
        rel=1e-15,
    )
    assert EARTH_TO_SOLAR_MASS_PARAMETER_RATIO == pytest.approx(
        NOMINAL_EARTH_MASS_PARAMETER_M3_S2 / NOMINAL_SOLAR_MASS_PARAMETER_M3_S2,
        rel=1e-15,
    )
    assert SOLAR_MEAN_DENSITY_G_CM3 == pytest.approx(1.4097798243, rel=1e-10)
    assert PARSEC_M == pytest.approx(3.085677581491367e16, rel=1e-15)


def test_kipping_and_claret_limb_darkening_coordinate_transforms():
    """
    NEDEN: (q1, q2) -> (u1, u2) -> (q1, q2) dönüşümünün tam tersinir olduğu ve
    Claret ızgarasındaki sınır durumlarda (homojen disk, lineer kararma) fiziki olmayan
    parlaklık değerleri üretmediği doğrulanır.
    """
    from exonym.lightcurve import (
        kipping_to_quadratic_limb_darkening,
        quadratic_to_kipping_limb_darkening,
    )

    payload = _load_fixture("claret_limb_darkening_sample.json")
    mu = np.linspace(0.0, 1.0, 1001)
    for case in payload["cases"]:
        q1, q2 = float(case["q1"]), float(case["q2"])
        expected_u = (float(case["u1"]), float(case["u2"]))
        u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
        assert (u1, u2) == pytest.approx(expected_u, abs=1e-14)
        recovered_q = quadratic_to_kipping_limb_darkening(u1, u2)
        expected_q = (0.0, 0.0) if q1 == 0.0 else (q1, q2)
        assert recovered_q == pytest.approx(expected_q, abs=1e-14)

        intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        d_intensity_d_mu = u1 + 2.0 * u2 * (1.0 - mu)
        assert np.min(intensity) >= -1e-14
        assert np.min(d_intensity_d_mu) >= -1e-14


@pytest.mark.parametrize("q1, q2", ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
def test_kipping_limb_darkening_cube_boundaries_are_finite_and_reversible(q1, q2):
    from exonym.lightcurve import (
        kipping_to_quadratic_limb_darkening,
        quadratic_to_kipping_limb_darkening,
    )

    u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
    assert np.isfinite((u1, u2)).all()
    expected = (0.0, 0.0) if q1 == 0.0 else (q1, q2)
    assert quadratic_to_kipping_limb_darkening(u1, u2) == pytest.approx(expected)


@pytest.mark.parametrize(
    "q1, q2",
    ((-1e-16, 0.5), (1.0 + 2e-16, 0.5), (float("nan"), 0.5), (0.5, float("inf"))),
)
def test_kipping_limb_darkening_rejects_out_of_cube_values(q1, q2):
    from exonym.lightcurve import kipping_to_quadratic_limb_darkening

    with pytest.raises(ValueError, match="must be in"):
        kipping_to_quadratic_limb_darkening(q1, q2)


@pytest.mark.parametrize("u1, u2", ((1.0, -1.0), (float("nan"), 0.0), (0.0, float("inf"))))
def test_quadratic_limb_darkening_rejects_invalid_inverse_coefficients(u1, u2):
    from exonym.lightcurve import quadratic_to_kipping_limb_darkening

    with pytest.raises(ValueError, match="coefficients"):
        quadratic_to_kipping_limb_darkening(u1, u2)


def test_mandel_agol_analytic_flux_against_quadrature_benchmarks():
    """
    NEDEN: Batman analitik ışık eğrisi çıktısının, referans çift katlı sayısal integrasyon
    sonuçlarıyla 1e-6 bağıl tolerans içinde eşleştiği kanıtlanarak gezegen yarıçapı
    hesabında kodlama/kütüphane hatası olmadığı belgelenir.
    """
    from exonym.transit_fit import batman_transit_flux

    payload = _load_fixture("mandel_agol_quadrature_benchmarks.json")
    tolerance = float(payload["quadrature"]["relative_tolerance"])
    for case in payload["cases"]:
        phases = np.asarray(case["phase_days"], dtype=float)
        expected = np.asarray(case["expected_flux"], dtype=float)
        quadrature = _quadrature_transit_flux(case, payload["quadrature"])
        analytic = batman_transit_flux(
            phases,
            case["period_days"],
            case["rp_rs"],
            case["a_rs"],
            case["impact_parameter"],
            case["q1"],
            case["q2"],
            case["baseline"],
            exposure_seconds=case["exposure_seconds"],
        )
        assert analytic is not None
        np.testing.assert_allclose(quadrature, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(analytic, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(analytic, quadrature, rtol=tolerance, atol=tolerance)


def test_keplerian_density_locking_semi_major_axis_calculation():
    """
    NEDEN: a/R_* ile Rp/R_* arasındaki güçlü korelasyonu kırmak için kullanılan
    (a/R_*)^3 = G*P^2*rho / (3*pi) formülünün CGS birim dönüşümlerinde sıfır hata ile
    çalıştığı doğrulanır.
    """
    from exonym.constants import GRAVITATIONAL_CONSTANT_CGS, SECONDS_PER_DAY, SOLAR_MEAN_DENSITY_G_CM3
    from exonym.transit_fit import stellar_density_a_rs

    cases = [
        (1.0, 365.25, 215.02944820799721),
        (1.0, 3.0, 8.7535717100598),
        (4.0, 3.0, 13.895428941028),
    ]
    for rho_solar, period_days, expected_a_rs in cases:
        expected_formula = (
            GRAVITATIONAL_CONSTANT_CGS
            * (period_days * SECONDS_PER_DAY) ** 2
            * (rho_solar * SOLAR_MEAN_DENSITY_G_CM3)
            / (3.0 * math.pi)
        ) ** (1.0 / 3.0)
        assert expected_formula == pytest.approx(expected_a_rs, rel=2e-4)
        assert stellar_density_a_rs(rho_solar, period_days) == pytest.approx(
            expected_formula, rel=1e-14
        )


def test_halley_kepler_solver_high_eccentricity_convergence():
    """
    NEDEN: e=0.95 gibi yüksek dışmerkezliklerde standart çözücüler ıraksayabilir.
    Halley 3. derece çözücüsünün 64 iterasyon içinde |E - e*sin(E) - M| <= 1e-12 radyan
    toleransına ulaştığı garanti altına alınır.
    """
    from exonym.radial_velocity import _solve_kepler_equation

    payload = _load_fixture("kepler_rv_eccentric_benchmarks.json")
    for case in payload["solver_cases"]:
        mean_anomaly = np.asarray(case["mean_anomaly_rad"], dtype=float)
        eccentricity = float(case["eccentricity"])
        eccentric_anomaly = _solve_kepler_equation(
            mean_anomaly,
            eccentricity,
            tolerance_rad=payload["solver_tolerance_rad"],
            max_iterations=payload["solver_max_iterations"],
        )
        expected = np.asarray(case["expected_eccentric_anomaly_rad"], dtype=float)
        independent = np.asarray(
            [_independent_eccentric_anomaly_rad(value, eccentricity) for value in mean_anomaly],
            dtype=float,
        )
        reduced_mean_anomaly = np.mod(mean_anomaly, _TAU)
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - reduced_mean_anomaly
        np.testing.assert_allclose(expected, independent, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(eccentric_anomaly, independent, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(residual)) <= payload["solver_tolerance_rad"]


def test_keplerian_rv_circular_and_eccentric_orbits():
    """
    NEDEN: Dairesel (e=0) yörüngede analitik kosinüs eğrisi ile sayısal Kepler çözücüsünün
    tam 0.0 m/s fark vermesi; dışmerkezli yörüngelerde ise yarı-genlik K'nın tam korunumu test edilir.
    """
    from exonym.catalog import calculate_radial_velocity_semi_amplitude
    from exonym.radial_velocity import keplerian_velocity_m_per_s

    payload = _load_fixture("kepler_rv_eccentric_benchmarks.json")
    for case in payload["rv_cases"]:
        actual = keplerian_velocity_m_per_s(
            case["time_bjd_tdb"],
            case["semi_amplitude_m_per_s"],
            case["mean_anomaly_reference_rad"],
            case["eccentricity"],
            case["argument_periastron_rad"],
            case["reference_time_bjd_tdb"],
            case["period_days"],
        )
        expected = np.asarray(case["expected_velocity_m_per_s"], dtype=float)
        independent = _independent_keplerian_velocity_m_per_s(case)
        np.testing.assert_allclose(expected, independent, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(actual, independent, rtol=0.0, atol=1e-11)

        eccentricity = float(case["eccentricity"])
        argument = float(case["argument_periastron_rad"])
        period = float(case["period_days"])
        reference_time = float(case["reference_time_bjd_tdb"])
        mean_reference = float(case["mean_anomaly_reference_rad"])
        if eccentricity == 0.0:
            mean_anomaly = mean_reference + _TAU * (
                np.asarray(case["time_bjd_tdb"]) - reference_time
            ) / period
            circular = float(case["semi_amplitude_m_per_s"]) * np.cos(mean_anomaly + argument)
            np.testing.assert_allclose(actual, circular, rtol=0.0, atol=1e-12)
        else:
            extrema_times = []
            for true_anomaly in (-argument, math.pi - argument):
                mean_anomaly = _true_anomaly_to_mean_anomaly(true_anomaly, eccentricity)
                phase = (mean_anomaly - mean_reference) % _TAU
                extrema_times.append(reference_time + period * phase / _TAU)
            extrema = keplerian_velocity_m_per_s(
                extrema_times,
                case["semi_amplitude_m_per_s"],
                mean_reference,
                eccentricity,
                argument,
                reference_time,
                period,
            )
            extrema_case = {**case, "time_bjd_tdb": extrema_times}
            independent_extrema = _independent_keplerian_velocity_m_per_s(extrema_case)
            np.testing.assert_allclose(extrema, independent_extrema, rtol=0.0, atol=1e-11)
            assert (float(np.max(extrema)) - float(np.min(extrema))) / 2.0 == pytest.approx(
                case["semi_amplitude_m_per_s"], rel=1e-12
            )

    amplitude_case = payload["semi_amplitude_case"]
    amplitude = calculate_radial_velocity_semi_amplitude(
        amplitude_case["m_planet_earth"],
        amplitude_case["m_star_solar"],
        amplitude_case["period_days"],
        amplitude_case["inclination_deg"],
        amplitude_case["eccentricity"],
    )
    assert amplitude == pytest.approx(amplitude_case["expected_k_m_per_s"], rel=1e-12)


def test_asteroseismic_scaling_relations_identity_and_giant_stars():
    """
    NEDEN: Güneş parametreleri girildiğinde M=1.0000 ve R=1.0000 çıkması (Identity Check);
    kırmızı dev parametrelerinde ise analitik formülün beklenen büyüme oranını verdiği doğrulanır.
    """
    from exonym.asteroseismology import seismic_mass_radius

    payload = _load_fixture("seismic_scaling_standard_stars.json")
    for case in payload["cases"]:
        result = seismic_mass_radius(case["numax_uhz"], case["dnu_uhz"], case["teff_k"])
        numax_ratio = float(case["numax_uhz"]) / 3090.0
        dnu_ratio = float(case["dnu_uhz"]) / 135.1
        teff_ratio = float(case["teff_k"]) / 5772.0
        expected_radius = numax_ratio * math.sqrt(teff_ratio) / dnu_ratio**2
        expected_mass = expected_radius**3 * dnu_ratio**2
        assert result["radius_solar"] == pytest.approx(case["expected_radius_solar"], abs=5e-4)
        assert result["mass_solar"] == pytest.approx(case["expected_mass_solar"], abs=5e-4)
        assert result["radius_solar"] == pytest.approx(round(expected_radius, 4), abs=5e-4)
        assert result["mass_solar"] == pytest.approx(round(expected_mass, 4), abs=5e-4)


def test_sed_blackbody_planck_radiation_and_extinction_scaling():
    """
    NEDEN: Planck fonksiyonunun sayısal taşma (overflow/underflow) yapmadığı, mesafe 2 katına
    çıktığında akının tam 4 kat azaldığı (1/d^2 kanunu) ve A_V=0 sönümlemesiz durum test edilir.
    """
    from exonym.sed import BAND_ZERO_POINTS, EXTINCTION_RATIOS, blackbody_model_magnitudes

    payload = _load_fixture("mist_synthetic_stellar_nodes.json")
    bands = payload["supported_planck_bands"]
    bandpasses = payload["planck_bandpasses"]
    assert set(bands) == set(BAND_ZERO_POINTS) == set(bandpasses)
    for band in bands:
        expected_bandpass = bandpasses[band]
        assert BAND_ZERO_POINTS[band] == pytest.approx(
            (
                expected_bandpass["pivot_wavelength_micron"],
                expected_bandpass["vega_zero_point_jy"],
            ),
            rel=0.0,
            abs=0.0,
        )
    for band, expected_ratio in payload["extinction_ratios"].items():
        assert EXTINCTION_RATIOS[band] == pytest.approx(expected_ratio, abs=0.0)

    for node in payload["nodes"]:
        band_data = [
            (
                band,
                bandpasses[band]["pivot_wavelength_micron"],
                bandpasses[band]["vega_zero_point_jy"],
            )
            for band in bands
        ]
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            no_extinction = blackbody_model_magnitudes(
                node["teff_k"], node["log_radius_over_distance"], 0.0, band_data
            )
            model = blackbody_model_magnitudes(
                node["teff_k"], node["log_radius_over_distance"], node["av_mag"], band_data
            )
        expected = np.asarray(
            [node["planck_reference_magnitudes"][band] for band in bands], dtype=float
        )
        independent_no_extinction = _independent_blackbody_magnitudes(
            {**node, "av_mag": 0.0}, bands, bandpasses, payload["extinction_ratios"]
        )
        independent_model = _independent_blackbody_magnitudes(
            node, bands, bandpasses, payload["extinction_ratios"]
        )
        assert set(node["mist_absolute_magnitudes"]) >= {
            "gaia_g",
            "gaia_bp",
            "gaia_rp",
            "twomass_j",
            "twomass_h",
            "twomass_ks",
            "wise_w1",
            "wise_w2",
        }
        assert all(
            math.isfinite(float(value))
            for value in node["mist_absolute_magnitudes"].values()
        )
        np.testing.assert_allclose(expected, independent_no_extinction, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(no_extinction, independent_no_extinction, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(model, independent_model, rtol=0.0, atol=1e-11)
        assert np.all(np.isfinite(model))
        assert np.all(np.isfinite(no_extinction))

    node = payload["nodes"][0]
    band_data = [
        (
            band,
            bandpasses[band]["pivot_wavelength_micron"],
            bandpasses[band]["vega_zero_point_jy"],
        )
        for band in bands
    ]
    near = blackbody_model_magnitudes(
        node["teff_k"], node["log_radius_over_distance"], 0.0, band_data
    )
    twice_distance = blackbody_model_magnitudes(
        node["teff_k"], node["log_radius_over_distance"] - math.log(2.0), 0.0, band_data
    )
    flux_ratio = 10.0 ** (-0.4 * twice_distance) / 10.0 ** (-0.4 * near)
    np.testing.assert_allclose(flux_ratio, 0.25, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(twice_distance - near, 5.0 * math.log10(2.0), rtol=0.0, atol=1e-12)

    extincted = blackbody_model_magnitudes(
        node["teff_k"], node["log_radius_over_distance"], 1.0, band_data
    )
    np.testing.assert_allclose(
        extincted - near,
        np.asarray([payload["extinction_ratios"][band] for band in bands]),
        rtol=0.0,
        atol=1e-12,
    )


def test_ttv_first_order_mean_motion_resonance_super_periods():
    """
    NEDEN: 2:1, 3:2, 4:3 ve 5:4 rezonanslarına yakın gezegen çiftlerinde Lithwick süper-periyot formülünün
    analitik olarak tam eşleştiği teyit edilir.
    """
    from exonym.search import calculate_ttv_super_period

    payload = _load_fixture("ttv_first_order_mmr_benchmarks.json")
    assert {case["j_resonance"] for case in payload["cases"]} >= {2, 3, 4, 5}
    for case in payload["cases"]:
        result = calculate_ttv_super_period(
            case["period_inner_days"], case["period_outer_days"], case["j_resonance"]
        )
        if case["expected_super_period_days"] == "infinity":
            assert math.isinf(result)
        else:
            expected = 1.0 / abs(
                case["j_resonance"] / case["period_outer_days"]
                - (case["j_resonance"] - 1) / case["period_inner_days"]
            )
            assert result == pytest.approx(case["expected_super_period_days"], rel=1e-12)
            assert result == pytest.approx(expected, rel=1e-12)


def test_phasecurve_circular_harmonics_and_cluster_covariance():
    """Check the circular BEER basis against an independently constructed regression table."""
    from exonym.phasecurve import (
        build_design_matrix,
        cluster_sandwich_covariance,
        fit_phase_curve_components,
    )

    payload = _load_fixture("phasecurve_circular_harmonic_benchmarks.json")
    case = payload["circular_case"]
    ephemeris = case["ephemeris"]
    secondary = case["secondary_box"]
    period_days = float(ephemeris["period_days"])
    epoch_btjd = float(ephemeris["epoch_btjd"])
    duration_days = float(ephemeris["duration_days"])

    anchor_phase_fractions = np.asarray(
        [anchor["phase_fraction"] for anchor in case["basis_anchors"]], dtype=float
    )
    anchor_phase_days = ((anchor_phase_fractions + 0.5) % 1.0 - 0.5) * period_days
    anchor_design, anchor_names, _ = build_design_matrix(
        np.arange(anchor_phase_days.size, dtype=float),
        anchor_phase_days,
        period_days,
        duration_days,
        np.ones(anchor_phase_days.size, dtype=int),
        block_days=float(case["block_days"]),
        secondary_eclipse_phase=float(secondary["phase_fraction"]),
        secondary_eclipse_duration_days=float(secondary["duration_days"]),
    )
    for row_index, anchor in enumerate(case["basis_anchors"]):
        for name, expected in anchor["expected_physical_columns"].items():
            assert anchor_design[row_index, anchor_names.index(name)] == pytest.approx(
                expected, rel=0.0, abs=2e-15
            )

    boundary_fractions = np.asarray(
        [anchor["phase_fraction"] for anchor in case["box_boundary_anchors"]], dtype=float
    )
    boundary_phase_days = ((boundary_fractions + 0.5) % 1.0 - 0.5) * period_days
    boundary_design, boundary_names, _ = build_design_matrix(
        np.arange(boundary_phase_days.size, dtype=float),
        boundary_phase_days,
        period_days,
        duration_days,
        np.ones(boundary_phase_days.size, dtype=int),
        block_days=float(case["block_days"]),
        secondary_eclipse_phase=float(secondary["phase_fraction"]),
        secondary_eclipse_duration_days=float(secondary["duration_days"]),
    )
    np.testing.assert_array_equal(
        boundary_design[:, boundary_names.index("secondary_eclipse_depth")],
        np.asarray(
            [anchor["expected_secondary_column"] for anchor in case["box_boundary_anchors"]],
            dtype=float,
        ),
    )

    sampling = case["sampling"]
    phase_numerators = np.arange(
        int(sampling["phase_numerator_start"]),
        int(sampling["phase_numerator_stop_inclusive"]) + 1,
        dtype=int,
    )
    phase_fractions = phase_numerators / float(sampling["phase_denominator"])
    local_fractions = np.concatenate(
        [phase_fractions + float(offset) for offset in sampling["orbit_offsets_days"]]
    )
    folded_fractions = local_fractions % 1.0
    angle = _TAU * folded_fractions
    eclipse = (
        np.abs(
            ((folded_fractions - float(secondary["phase_fraction"]) + 0.5) % 1.0) - 0.5
        )
        < float(secondary["strict_half_width_phase"])
    ).astype(float)
    independent_components = {
        "reflection_semiamplitude": -np.cos(angle),
        "beaming_semiamplitude": np.sin(angle),
        "ellipsoidal_semiamplitude": -np.cos(2.0 * angle),
        "second_harmonic_sine_control": np.sin(2.0 * angle),
        "secondary_eclipse_depth": -eclipse,
    }
    time_parts = []
    flux_parts = []
    sector_parts = []
    for sector in sampling["sectors"]:
        sector_time = float(sector["start_btjd"]) + local_fractions
        sector_flux = (
            float(sector["offset_relative_flux"])
            + float(sector["slope_relative_flux_per_day"])
            * (sector_time - float(np.median(sector_time)))
        )
        for name, basis in independent_components.items():
            sector_flux += float(case["injected_component_ppm"][name]) * 1e-6 * basis
        time_parts.append(sector_time)
        flux_parts.append(sector_flux)
        sector_parts.append(np.full(sector_time.size, int(sector["label"]), dtype=int))
    time = np.concatenate(time_parts)
    flux = np.concatenate(flux_parts)
    sectors = np.concatenate(sector_parts)
    flux_err = np.full(time.size, float(sampling["flux_err_relative_flux"]), dtype=float)
    phase_days = ((time - epoch_btjd + 0.5 * period_days) % period_days) - 0.5 * period_days

    design, names, cluster = build_design_matrix(
        time,
        phase_days,
        period_days,
        duration_days,
        sectors,
        block_days=float(case["block_days"]),
        secondary_eclipse_phase=float(secondary["phase_fraction"]),
        secondary_eclipse_duration_days=float(secondary["duration_days"]),
    )
    expected = case["expected"]
    expected_cluster_per_sector = np.repeat(
        np.arange(len(expected["cluster_run_lengths_per_sector"]), dtype=int),
        expected["cluster_run_lengths_per_sector"],
    )
    expected_cluster = np.concatenate(
        [
            expected_cluster_per_sector,
            expected_cluster_per_sector + int(expected["second_sector_cluster_offset"]),
        ]
    )
    np.testing.assert_array_equal(cluster, expected_cluster)
    expected_eclipse = -np.isin(
        np.tile(phase_numerators, len(sampling["orbit_offsets_days"])),
        expected["box_interior_phase_numerators"],
    ).astype(float)
    np.testing.assert_array_equal(
        design[:, names.index("secondary_eclipse_depth")],
        np.tile(expected_eclipse, len(sampling["sectors"])),
    )

    result = fit_phase_curve_components(
        time,
        flux,
        flux_err,
        sectors,
        ephemeris,
        block_days=float(case["block_days"]),
        primary_mask_half_durations=float(case["primary_mask_half_durations"]),
        secondary_eclipse_phase=float(secondary["phase_fraction"]),
        secondary_eclipse_duration_days=float(secondary["duration_days"]),
    )
    assert result["n_points_after_primary_transit_mask"] == expected["retained_points"]
    assert result["n_sectors"] == expected["sector_count"]
    assert result["n_covariance_clusters"] == expected["cluster_count"]
    assert result["secondary_box_template_method"] == expected["secondary_box_template_method"]
    assert result["secondary_box_duration_hours"] == pytest.approx(
        expected["secondary_box_duration_hours"], abs=0.0
    )
    for name, value_ppm in expected["component_values_ppm"].items():
        assert result["components"][name]["value_ppm"] == pytest.approx(value_ppm, abs=0.001)

    covariance_case = payload["cluster_sandwich_case"]
    covariance, n_clusters = cluster_sandwich_covariance(
        np.asarray(covariance_case["design"], dtype=float),
        np.asarray(covariance_case["residual"], dtype=float),
        np.asarray(covariance_case["sigma"], dtype=float),
        np.asarray(covariance_case["cluster"], dtype=int),
    )
    assert n_clusters == covariance_case["expected_n_clusters"]
    np.testing.assert_allclose(
        covariance,
        np.asarray(covariance_case["expected_covariance"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


def test_cross_match_isochrone_evolution_consistency():
    """
    NEDEN: cross_match_isochrone_evolution fonksiyonunun Güneş parametrelerinde
    tam ana kol tespiti yaptığını, dev yıldızlarda subgiant_or_evolved sonucunu
    verdiğini ve geçersiz girdileri (negatif Teff / R) reddettiğini doğrular.
    """
    from exonym.sed import cross_match_isochrone_evolution
    from exonym.constants import (
        NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
        NOMINAL_SOLAR_LOGG_CGS,
    )

    # 1. Solar identity check -> main sequence
    solar = cross_match_isochrone_evolution(
        teff_k=NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
        logg_cgs=NOMINAL_SOLAR_LOGG_CGS,
        feh_dex=0.0,
        radius_solar=1.0,
    )
    assert solar["evolutionary_stage"] == "main_sequence"
    assert solar["teff_ratio_vs_solar"] == pytest.approx(1.0, rel=1e-4)
    assert solar["observed_logg_offset"] == pytest.approx(0.0, abs=1e-3)

    # 2. Red giant star (R=12.0 R_sun, logg=2.3) -> subgiant_or_evolved
    giant = cross_match_isochrone_evolution(
        teff_k=4500.0,
        logg_cgs=2.3,
        feh_dex=-0.1,
        radius_solar=12.0,
    )
    assert giant["evolutionary_stage"] == "subgiant_or_evolved"
    assert giant["observed_logg_offset"] > 0.4

    # 3. Non-physical input error checking
    with pytest.raises(ValueError, match="teff_k must be positive"):
        cross_match_isochrone_evolution(teff_k=-5000.0, logg_cgs=4.4, feh_dex=0.0, radius_solar=1.0)

    with pytest.raises(ValueError, match="radius_solar must be positive"):
        cross_match_isochrone_evolution(teff_k=5772.0, logg_cgs=4.4, feh_dex=0.0, radius_solar=-1.0)


def test_secular_timing_models_apsidal_precession_and_degeneracy():
    """
    Verify secular timing models (linear, decay, apsidal precession, Rømer LTT)
    and baseline coverage degeneracy warning (N_span * |omega_dot| < pi/2).
    """
    from exonym.ttv import fit_secular_timing_models

    period = 1.5
    t0 = 1200.0
    epochs = np.arange(0, 120, dtype=int)
    # Synthetic linear transits
    transits = t0 + epochs * period
    errors = np.full_like(transits, 0.0005)

    res = fit_secular_timing_models(epochs, transits, errors)
    assert res["status"] == "compared"
    assert "linear" in res["models"]
    assert "quadratic_decay" in res["models"]
    assert "apsidal_precession" in res["models"]
    assert "roemer_ltt" in res["models"]
    assert res["preferred_model_bic"] == "linear"
    assert res["preferred_model_aic"] == "linear"

    # Degeneracy check on short baseline
    short_epochs = np.arange(0, 8, dtype=int)
    short_transits = t0 + short_epochs * period
    short_errors = np.full_like(short_transits, 0.0005)
    res_short = fit_secular_timing_models(short_epochs, short_transits, short_errors)
    apsidal_mod = res_short["models"]["apsidal_precession"]
    assert apsidal_mod["baseline_coverage_warning"] is not None
    assert "insufficient to break mathematical degeneracy" in apsidal_mod["baseline_coverage_warning"]


def test_bayesian_model_comparison_nested_sampling_evidence():
    """
    Verify Bayesian model comparison between eccentric and circular hypotheses
    on the Kass & Raftery (1995) scale.
    """
    from exonym.transit_fit import compute_bayesian_model_comparison

    # Decisive evidence for eccentric: Delta ln Z = 6.0 >= 5.0
    comp_decisive = compute_bayesian_model_comparison(
        ln_z_circular=1000.0,
        ln_z_circular_err=0.1,
        ln_z_eccentric=1006.0,
        ln_z_eccentric_err=0.1,
    )
    assert comp_decisive["preferred_model"] == "eccentric"
    assert comp_decisive["kass_raftery_scale"] == "decisive_evidence"
    assert comp_decisive["delta_ln_z"] == pytest.approx(6.0, abs=1e-5)
    assert comp_decisive["bayes_factor"] == pytest.approx(math.exp(6.0), rel=1e-4)

    # Inconclusive evidence: Delta ln Z = 0.5 < 1.0
    comp_inconclusive = compute_bayesian_model_comparison(
        ln_z_circular=1000.0,
        ln_z_circular_err=0.1,
        ln_z_eccentric=1000.5,
        ln_z_eccentric_err=0.1,
    )
    assert comp_inconclusive["kass_raftery_scale"] == "inconclusive"

    # Substantial evidence for circular: Delta ln Z = -2.0
    comp_sub = compute_bayesian_model_comparison(
        ln_z_circular=1002.0,
        ln_z_circular_err=0.1,
        ln_z_eccentric=1000.0,
        ln_z_eccentric_err=0.1,
    )
    assert comp_sub["preferred_model"] == "circular"
    assert comp_sub["kass_raftery_scale"] == "substantial_evidence"


def test_harvey_convective_granulation_background_recovery():
    """
    Verify Harvey convective granulation background fitting (W + 2 Harvey components + p-mode envelope).
    """
    from exonym.asteroseismology import fit_harvey_granulation_background

    # Synthetic PSD grid
    nu = np.linspace(10.0, 2000.0, 500)
    w_true = 5.0
    a1_true = 120.0
    tau1_true = 20000.0
    a2_true = 35.0
    tau2_true = 700.0
    h_osc_true = 50.0
    numax_true = 800.0
    sigma_env_true = 100.0

    scale1 = 2.0 * math.pi * 1e-6 * tau1_true * nu
    scale2 = 2.0 * math.pi * 1e-6 * tau2_true * nu
    p_bg = w_true + a1_true / (1.0 + scale1**2) + a2_true / (1.0 + scale2**4)
    p_osc = h_osc_true * np.exp(-0.5 * ((nu - numax_true) / sigma_env_true)**2)
    p_total = p_bg + p_osc

    res = fit_harvey_granulation_background(nu, p_total, numax_guess=numax_true)
    assert res["status"] in ("converged", "optimization_suboptimal")
    assert res["white_noise_floor_w"] > 0.0
    assert res["timescale_tau1_seconds"] > 1000.0
    assert res["timescale_tau2_seconds"] > 10.0
    assert res["numax_uhz"] == pytest.approx(numax_true, rel=0.15)
    assert res["reduced_chi2"] >= 0.0
    assert len(res["whitened_power"]) == len(nu)
