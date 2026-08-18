import json
import io

import pytest

from exonym.catalog_federation import (
    PROVIDERS,
    CatalogRequest,
    TransportResponse,
    _request_for,
    catalog_report,
    fetch_catalog,
    normalize_cross_matches,
    refresh_catalog,
)
from exonym.__main__ import main
from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate, load_candidate


def _candidate(tmp_path):
    create_candidate(tmp_path, "catalog-target", tic="123456789", mission="tess")
    metadata_path = tmp_path / "candidate" / "catalog-target" / "candidate.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["identifiers"]["aliases"].append("Gaia DR3 123456789")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return load_candidate(tmp_path, "catalog-target")


def _transport(status_code, body, calls=None):
    def send(request):
        if calls is not None:
            calls.append(request)
        return TransportResponse(status_code, {"ETag": "synthetic"}, body)

    return send


def _write_coordinate_context(candidate):
    (candidate.path / "outputs" / "archival_vetting_report.json").write_text(
        json.dumps({"target_coordinates": {"ra_deg": 10.0, "dec_deg": 20.0}}),
        encoding="utf-8",
    )


def _provider_success_body(expected_format):
    if expected_format == "json":
        return b'{"data": [{"synthetic": "record"}]}'
    if expected_format == "csv":
        return b"name\nsynthetic\n"
    if expected_format in ("votable", "ipac-table"):
        from astropy.table import Table

        if expected_format == "votable":
            buffer = io.BytesIO()
            Table({"name": ["synthetic"]}).write(buffer, format="votable")
            return buffer.getvalue()
        buffer = io.StringIO()
        Table({"name": ["synthetic"]}).write(buffer, format="ascii.ipac")
        return buffer.getvalue().encode("utf-8")
    return b"<html><body>synthetic provider document</body></html>"


def _audit(root):
    report = IsolationReport()
    validate_schemas(root, report)
    return report


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_each_allowlisted_provider_uses_a_fixed_template_and_captures_success(tmp_path, provider):
    # Arrange
    candidate = _candidate(tmp_path)
    calls = []
    spec = PROVIDERS[provider]
    body = _provider_success_body(spec.expected_format)

    # Act
    manifest_path = fetch_catalog(candidate, [provider], _transport(200, body, calls))[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Assert
    assert manifest["provider"] == provider
    assert manifest["status"] == ("unavailable" if spec.requires_coordinates else "available")
    if spec.requires_coordinates:
        assert not calls
    else:
        assert calls[0].source_uri.startswith("https://")
    assert "credentials" not in manifest["request_parameters"]
    raw_dir = candidate.path / "data" / "external" / "catalog" / provider / manifest["retrieval_id"]
    assert (raw_dir / "response.bin").read_bytes() == (b"" if spec.requires_coordinates else body)
    assert (candidate.path / "outputs" / "catalog_context.json").is_file()
    assert _audit(tmp_path).ok


@pytest.mark.parametrize("provider", ["lamost-dr11", "smoka", "mast-hubble-jwst"])
def test_coordinate_bound_discovery_templates_are_fixed_and_metadata_only(tmp_path, provider):
    # Arrange
    candidate = _candidate(tmp_path)
    _write_coordinate_context(candidate)
    calls = []
    spec = PROVIDERS[provider]

    # Act
    body = (
        b'{"data": [{"obs_collection": "HST"}, {"obs_collection": "other"}]}'
        if provider == "mast-hubble-jwst" else _provider_success_body(spec.expected_format)
    )
    manifest_path = fetch_catalog(candidate, [provider], _transport(200, body, calls))[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Assert
    assert manifest["status"] == "available"
    assert calls[0].method == "POST"
    assert calls[0].source_uri in {
        "https://www.lamost.org/dr11/v2.0/table/combined/q",
        "https://smoka.nao.ac.jp/fssearch",
        "https://mast.stsci.edu/api/v0/invoke",
    }
    assert "download" not in calls[0].source_uri
    assert "authorization" not in {key.lower() for key in calls[0].headers}
    cross_match = json.loads(manifest_path.with_name("cross-match.json").read_text(encoding="utf-8"))
    assert cross_match["cross_match"]["status"] == "insufficient-astrometry"
    if provider == "mast-hubble-jwst":
        discovery = json.loads(manifest_path.with_name("archive-discovery.json").read_text(encoding="utf-8"))
        assert [row["source_record_index"] for row in discovery["records"]] == [0]
    assert _audit(tmp_path).ok


def test_cross_match_propagates_proper_motion_and_retains_ambiguous_matches():
    # Arrange
    target = {
        "ra_deg": 10.0, "dec_deg": 0.0, "reference_epoch_jyear": 2000.0,
        "ra_uncertainty_mas": 10.0, "dec_uncertainty_mas": 10.0,
    }
    sources = [
        {
            "ra_deg": 10.001, "dec_deg": 0.0, "reference_epoch_jyear": 2016.0,
            "pmra_mas_per_year": 225.0, "pmdec_mas_per_year": 0.0,
            "ra_uncertainty_mas": 10.0, "dec_uncertainty_mas": 10.0,
        },
        {
            "ra_deg": 10.0, "dec_deg": 0.000001, "reference_epoch_jyear": 2000.0,
            "pmra_mas_per_year": 0.0, "pmdec_mas_per_year": 0.0,
            "ra_uncertainty_mas": 10.0, "dec_uncertainty_mas": 10.0,
        },
    ]

    # Act
    result = normalize_cross_matches(target, sources)

    # Assert
    assert result["status"] == "ambiguous"
    assert [match["source_record_index"] for match in result["matches"]] == [0, 1]


def test_catalog_normalization_rejects_declared_demonstration_values(tmp_path):
    # Arrange
    candidate = _candidate(tmp_path)
    _write_coordinate_context(candidate)
    body = b"ra,dec,source,mass_solar,radius_solar\n10,20,synthetic-demo,1.0,1.0\n"

    # Act
    manifest = fetch_catalog(candidate, ["lamost-dr11"], _transport(200, body))[0]
    parameters = json.loads(manifest.with_name("stellar-parameters.json").read_text(encoding="utf-8"))

    # Assert
    assert parameters["records"] == []


def test_catalog_records_empty_response_without_a_novelty_or_claim(tmp_path):
    # Arrange
    candidate = _candidate(tmp_path)

    # Act
    manifest_path = fetch_catalog(candidate, ["mast"], _transport(200, b'{"data": []}'))[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Assert
    assert manifest["status"] == "empty"
    assert not list((candidate.path / "claims").iterdir())
    assert candidate.metadata["workflow"]["phase"] == "intake"


def test_catalog_records_unavailable_and_malformed_responses(tmp_path):
    # Arrange
    candidate = _candidate(tmp_path)

    # Act
    unavailable = fetch_catalog(candidate, ["mast"], _transport(503, b"offline"))[0]
    malformed = fetch_catalog(candidate, ["mast"], _transport(200, b"not-json"))[0]

    # Assert
    assert json.loads(unavailable.read_text(encoding="utf-8"))["status"] == "unavailable"
    assert json.loads(malformed.read_text(encoding="utf-8"))["status"] == "unavailable"
    log = malformed.with_name("parser-log.json")
    assert "malformed" in json.loads(log.read_text(encoding="utf-8"))["message"]


@pytest.mark.parametrize("body", [b'{"data": [{"value": NaN}]}', b'{"data": [{"value": Infinity}]}'])
def test_catalog_rejects_nonfinite_provider_json(tmp_path, body):
    candidate = _candidate(tmp_path)

    manifest_path = fetch_catalog(candidate, ["mast"], _transport(200, body))[0]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "unavailable"


def test_catalog_records_ambiguous_identity_matches_and_authentication(tmp_path):
    # Arrange
    candidate = _candidate(tmp_path)

    # Act
    ambiguous = fetch_catalog(candidate, ["mast"], _transport(200, b'{"data": [{}, {}]}'))[0]
    authentication = fetch_catalog(candidate, ["mast"], _transport(401, b"auth"))[0]

    # Assert
    assert json.loads(ambiguous.read_text(encoding="utf-8"))["status"] == "ambiguous"
    assert json.loads(authentication.read_text(encoding="utf-8"))["status"] == "requires-authentication"
    report = catalog_report(candidate)
    assert any(record["status"] == "ambiguous" for record in report["retrievals"])


def test_catalog_retries_retryable_failures_and_refresh_is_append_only(tmp_path):
    # Arrange
    candidate = _candidate(tmp_path)
    responses = [
        TransportResponse(503, {}, b"temporary"),
        TransportResponse(200, {}, b'{"data": [{"ok": true}]}'),
    ]
    calls = []

    def transport(request):
        calls.append(request)
        return responses.pop(0)

    # Act
    manifest = fetch_catalog(candidate, ["mast"], transport)[0]
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["expires_at"] = "2000-01-01T00:00:00Z"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    refreshed = refresh_catalog(candidate, _transport(200, b'{"data": [{"fresh": true}]}'))

    # Assert
    assert len(calls) == 2
    assert record["attempts"] == 2
    assert len(refreshed) == 1
    assert refreshed[0] != manifest
    assert manifest.is_file()


def test_gaia_template_requires_a_recorded_source_identifier(tmp_path):
    candidate = _candidate(tmp_path)

    request = _request_for(PROVIDERS["gaia"], candidate)

    assert request.method == "GET"
    assert "SELECT" in request.source_uri
    assert "123456789" in request.parameters["gaia_source_id"]


def test_catalog_schema_rejects_mismatched_snapshot_ownership(tmp_path):
    # Arrange
    candidate = _candidate(tmp_path)
    manifest = fetch_catalog(candidate, ["mast"], _transport(200, b'{"data": [{}]}'))[0]
    snapshot = manifest.with_name("snapshot.json")
    record = json.loads(snapshot.read_text(encoding="utf-8"))
    record["candidate_id"] = "other-candidate"
    snapshot.write_text(json.dumps(record), encoding="utf-8")

    # Act
    report = _audit(tmp_path)

    # Assert
    assert any(violation.path == snapshot.as_posix() for violation in report.violations)


def test_catalog_cli_dispatches_only_allowlisted_provider_names(tmp_path, monkeypatch, capsys):
    # Arrange
    candidate = _candidate(tmp_path)
    called = []

    def fake_fetch(workspace, providers):
        called.append((workspace.candidate_id, providers))
        return []

    monkeypatch.setattr("exonym.catalog_federation.fetch_catalog", fake_fetch)

    # Act
    result = main(["--root", str(tmp_path), "catalog", "fetch", candidate.candidate_id, "--providers", "mast"])

    # Assert
    assert result == 0
    assert called == [(candidate.candidate_id, ["mast"])]
    assert "[]" in capsys.readouterr().out


def test_freeze_indexes_catalog_retrieval_manifests(tmp_path):
    # Arrange
    from exonym.freeze import freeze

    candidate = _candidate(tmp_path)
    retrieval = fetch_catalog(candidate, ["mast"], _transport(200, b'{"data": [{}]}'))[0]
    (tmp_path / "requirements-lock.txt").write_text("numpy==1.26.4\n", encoding="utf-8")

    # Act
    release = freeze(candidate, version="v1.0.0")
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))

    # Assert
    assert len(manifest["catalog_manifests"]) == 1
    assert manifest["catalog_manifests"][0]["manifest_path"] == retrieval.relative_to(candidate.path).as_posix()
