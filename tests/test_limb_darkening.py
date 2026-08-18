import hashlib
import json
import sys
import types

import pytest

from exonym.limb_darkening import OUTPUT_FILENAME, generate_ldtk_quadratic_prior


class _Filter:
    def __init__(self, name):
        self.name = name


def _workspace(tmp_path):
    parameters_path = tmp_path / "data" / "external" / "stellar_params.json"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(
        json.dumps(
            {
                "teff_k": 5700.0,
                "teff_err_k": 75.0,
                "logg_cgs": 4.3,
                "logg_err_cgs": 0.1,
                "feh": -0.1,
                "feh_err": 0.05,
            }
        ),
        encoding="utf-8",
    )
    return types.SimpleNamespace(path=tmp_path, candidate_id="synthetic-star"), parameters_path


def _fake_ldtk(calls):
    class Profiles:
        def coeffs_qd(self, do_mc):
            calls["do_mc"] = do_mc
            return [[0.2, 0.3]], [[0.01, 0.02]]

    class Creator:
        def __init__(self, **kwargs):
            calls["creator"] = kwargs

        def create_profiles(self):
            return Profiles()

    module = types.ModuleType("ldtk")
    module.__version__ = "mocked-version"
    module.LDPSetCreator = Creator
    return module


def test_generate_ldtk_prior_writes_candidate_local_provenance(tmp_path, monkeypatch):
    # Arrange
    workspace, parameters_path = _workspace(tmp_path)
    calls = {}
    monkeypatch.setitem(sys.modules, "ldtk", _fake_ldtk(calls))

    # Act
    output_path = generate_ldtk_quadratic_prior(workspace, [_Filter("synthetic-band")])

    # Assert
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path == tmp_path / "outputs" / OUTPUT_FILENAME
    assert calls["creator"]["teff"] == (5700.0, 75.0)
    assert calls["creator"]["logg"] == (4.3, 0.1)
    assert calls["creator"]["z"] == (-0.1, 0.05)
    assert calls["do_mc"] is True
    assert payload["ldtk"]["version"] == "mocked-version"
    assert payload["input_provenance"]["stellar_parameters_sha256"] == hashlib.sha256(
        parameters_path.read_bytes()
    ).hexdigest()
    assert payload["quadratic_coefficients"] == [
        {
            "filter": "synthetic-band",
            "u1": 0.2,
            "u1_err": 0.01,
            "u2": 0.3,
            "u2_err": 0.02,
            "unit": "dimensionless",
        }
    ]


def test_generate_ldtk_prior_rejects_missing_uncertainty_without_writing(tmp_path, monkeypatch):
    # Arrange
    workspace, parameters_path = _workspace(tmp_path)
    payload = json.loads(parameters_path.read_text(encoding="utf-8"))
    del payload["feh_err"]
    parameters_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "ldtk", _fake_ldtk({}))

    # Act and assert
    with pytest.raises(ValueError, match="feh uncertainty"):
        generate_ldtk_quadratic_prior(workspace, [_Filter("synthetic-band")])
    assert not (tmp_path / "outputs" / OUTPUT_FILENAME).exists()


def test_generate_ldtk_prior_rejects_nonfinite_stellar_parameters_without_writing(tmp_path, monkeypatch):
    # Arrange
    workspace, parameters_path = _workspace(tmp_path)
    payload = json.loads(parameters_path.read_text(encoding="utf-8"))
    payload["logg_cgs"] = float("nan")
    parameters_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "ldtk", _fake_ldtk({}))

    # Act and assert
    with pytest.raises(ValueError, match="logg_cgs must be a finite number"):
        generate_ldtk_quadratic_prior(workspace, [_Filter("synthetic-band")])
    assert not (tmp_path / "outputs" / OUTPUT_FILENAME).exists()


def test_generate_ldtk_prior_raises_when_dependency_is_unavailable(tmp_path, monkeypatch):
    # Arrange
    workspace, _ = _workspace(tmp_path)
    monkeypatch.setitem(sys.modules, "ldtk", None)

    # Act and assert
    with pytest.raises(RuntimeError, match="dependency is unavailable"):
        generate_ldtk_quadratic_prior(workspace, [_Filter("synthetic-band")])
    assert not (tmp_path / "outputs" / OUTPUT_FILENAME).exists()
