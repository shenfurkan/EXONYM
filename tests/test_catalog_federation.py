import json
import io
from urllib.parse import unquote_plus

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
from exonym.ephemeris_matching import (
    match_known_signal_ephemerides,
    record_known_signal_ephemeris,
)
from exonym.__main__ import main
from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate, load_candidate


def _candidate(tmp_path):
    package_dir = tmp_path / "src" / "exonym"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text('__version__ = "test"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = [\"setuptools\"]\nbuild-backend = \"setuptools.build_meta\"\n"
        "[project]\nname = \"exonym\"\nversion = \"0.0.0\"\n",
        encoding="utf-8",
    )
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


def _write_candidate_ephemeris(candidate, period_days=3.0, epoch_btjd=1.0, duration_hours=2.0):
    (candidate.path / "config" / "transit_config.json").write_text(
        json.dumps(
            {
                "transit": {
                    "period_days": period_days,
                    "epoch_btjd": epoch_btjd,
                    "duration_hours": duration_hours,
                }
            }
        ),
        encoding="utf-8",
    )


def _provider_success_body(spec):
    if spec.name == "nasa-exoplanet-archive":
        return b"pl_name,pl_orbper,pl_tranmid,pl_trandur,pl_tranmid_systemref\nSynthetic b,3.0,2457001.0,2.0,BJD_TDB\n"
    if spec.name == "nasa-exoplanet-archive-toi":
        return b"toi,tid,pl_orbper,pl_tranmid,pl_trandurh\n100.01,123456789,3.0,2457001.0,2.0\n"
    if spec.expected_format == "json":
        return b'{"data": [{"synthetic": "record"}]}'
    if spec.expected_format == "csv":
        return b"name\nsynthetic\n"
    if spec.expected_format in ("votable", "ipac-table"):
        from astropy.table import Table

        if spec.expected_format == "votable":
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
    body = _provider_success_body(spec)

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


def test_nasa_known_signal_templates_use_current_tic_contracts(tmp_path):
    candidate = _candidate(tmp_path)

    confirmed = _request_for(PROVIDERS["nasa-exoplanet-archive"], candidate)
    toi = _request_for(PROVIDERS["nasa-exoplanet-archive-toi"], candidate)

    assert "pl_tranmid_systemref" in unquote_plus(confirmed.source_uri)
    assert "tic_id = 'TIC 123456789'" in unquote_plus(confirmed.source_uri)
    assert "SELECT toi,tid" in unquote_plus(toi.source_uri)
    assert "WHERE tid = 123456789" in unquote_plus(toi.source_uri)


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
        if provider == "mast-hubble-jwst" else _provider_success_body(spec)
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


def test_known_signal_match_is_hash_bound_and_review_required(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    body = (
        b"pl_orbper,pl_tranmid,pl_trandur,pl_tranmid_systemref,pl_name\n"
        b"3.0,2457001.0,2.0,BJD_TDB,Known Planet b\n"
    )
    fetch_catalog(candidate, ["nasa-exoplanet-archive"], _transport(200, body))

    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["status"] == "review-required-known-signal-match"
    assert record["comparisons"][0]["period_harmonic_match"] is True
    assert record["comparisons"][0]["known_epoch_time_scale"] == "BJD_TDB"
    assert record["comparisons"][0]["epoch_match"] is True
    assert record["comparisons"][0]["review_required"] is True
    assert record["source_snapshots"][0]["snapshot"]["sha256"]
    assert _audit(tmp_path).ok


def test_known_signal_toi_retrieval_limits_epoch_matching_without_bjd_tdb_contract(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    body = b"toi,tid,pl_orbper,pl_tranmid,pl_trandurh\n100.01,123456789,3.0,2457001.0,2.0\n"
    fetch_catalog(candidate, ["nasa-exoplanet-archive-toi"], _transport(200, body))

    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["status"] == "review-required-period-harmonic"
    assert record["configuration"]["supported_providers"] == [
        "nasa-exoplanet-archive", "nasa-exoplanet-archive-toi", "candidate-recorded-evidence"
    ]
    assert record["source_snapshots"][0]["provider"] == "nasa-exoplanet-archive-toi"
    comparison = record["comparisons"][0]
    assert comparison["known_epoch_bjd_tdb"] is None
    assert comparison["known_epoch_time_scale"] == "BJD_UNSPECIFIED"
    assert comparison["epoch_match"] is None
    assert comparison["duration_compatible"] is True
    assert comparison["review_required"] is True
    assert _audit(tmp_path).ok


def test_known_signal_parser_rejects_missing_provider_contract_columns(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    body = b"pl_orbper,pl_tranmid,pl_trandur\n3.0,2457001.0,2.0\n"

    manifest_path = fetch_catalog(candidate, ["nasa-exoplanet-archive"], _transport(200, body))[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parser_log = json.loads(manifest_path.with_name("parser-log.json").read_text(encoding="utf-8"))
    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["status"] == "unavailable"
    assert parser_log["known_signal_required_columns"] == [
        "pl_orbper", "pl_tranmid", "pl_trandur", "pl_tranmid_systemref"
    ]
    assert "known-signal field contract missing columns" in parser_log["message"]
    assert record["status"] == "insufficient-current-supported-catalog-evidence"
    assert record["comparisons"] == []


def test_known_signal_match_never_treats_absent_supported_retrievals_as_novelty(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)

    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["status"] == "insufficient-current-supported-catalog-evidence"
    assert record["comparisons"] == []
    assert "does not establish novelty" in record["limitations"]


def test_known_signal_match_no_match_is_limited_to_the_current_snapshot(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    body = b"pl_orbper,pl_tranmid,pl_trandur,pl_tranmid_systemref\n5.0,2457001.0,2.0,BJD_TDB\n"
    fetch_catalog(candidate, ["nasa-exoplanet-archive"], _transport(200, body))

    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["status"] == "no-ephemeris-match-in-current-supported-catalog"
    assert record["comparisons"][0]["review_required"] is False
    assert "does not establish novelty" in record["limitations"]


def test_known_signal_match_excludes_stale_retrievals(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    body = b"pl_orbper,pl_tranmid,pl_trandur,pl_tranmid_systemref\n3.0,2457001.0,2.0,BJD_TDB\n"
    manifest_path = fetch_catalog(candidate, ["nasa-exoplanet-archive"], _transport(200, body))[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expires_at"] = "2000-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["status"] == "insufficient-current-supported-catalog-evidence"
    assert record["comparisons"] == []
    assert record["excluded_retrievals"][0]["reason"] == "retrieval-stale-or-invalid-time"


def test_known_signal_match_schema_rejects_tampered_snapshot_binding(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    body = b"pl_orbper,pl_tranmid,pl_trandur,pl_tranmid_systemref\n3.0,2457001.0,2.0,BJD_TDB\n"
    manifest = fetch_catalog(candidate, ["nasa-exoplanet-archive"], _transport(200, body))[0]
    match_known_signal_ephemerides(candidate)
    snapshot = manifest.with_name("snapshot.json")
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = _audit(tmp_path)

    assert any(
        violation.rule == "artifact-hash-mismatch" and "known-signal catalog inputs" in violation.detail
        for violation in report.violations
    )


def test_recorded_known_signal_evidence_is_hash_bound_and_requires_review(tmp_path):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)
    raw_artifact = candidate.path / "literature" / "known-eb-row.txt"
    raw_artifact.write_text("reviewed source row", encoding="utf-8")

    evidence = record_known_signal_ephemeris(
        candidate,
        "eb-row-01",
        "eclipsing-binary-catalog",
        "Reviewed known binary",
        "https://example.invalid/known-binary",
        "literature/known-eb-row.txt",
        3.0,
        2457001.0,
        2.0,
        "2026-08-01T00:00:00Z",
        "2026-09-01T00:00:00Z",
    )

    output = match_known_signal_ephemerides(candidate)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert evidence == candidate.path / "decisions" / "known_signal_ephemerides.json"
    assert record["status"] == "review-required-known-signal-match"
    assert record["source_snapshots"][0]["provider"] == "candidate-recorded-evidence"
    assert record["comparisons"][0]["provider"] == "candidate-recorded-evidence"
    assert _audit(tmp_path).ok


def test_recorded_known_signal_evidence_rejects_unbound_or_tampered_inputs(tmp_path):
    candidate = _candidate(tmp_path)
    raw_artifact = candidate.path / "literature" / "known-variable-row.txt"
    raw_artifact.write_text("reviewed source row", encoding="utf-8")

    with pytest.raises(ValueError, match="BJD_TDB"):
        record_known_signal_ephemeris(
            candidate,
            "variable-row-01",
            "variable-star-catalog",
            "Reviewed variable",
            "https://example.invalid/variable",
            "literature/known-variable-row.txt",
            3.0,
            float("nan"),
            2.0,
            "2026-08-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        )

    evidence = record_known_signal_ephemeris(
        candidate,
        "variable-row-01",
        "variable-star-catalog",
        "Reviewed variable",
        "https://example.invalid/variable",
        "literature/known-variable-row.txt",
        3.0,
        2457001.0,
        2.0,
        "2026-08-01T00:00:00Z",
        "2026-09-01T00:00:00Z",
    )
    raw_artifact.write_text("tampered source row", encoding="utf-8")

    report = _audit(tmp_path)

    assert evidence.is_file()
    assert any(
        violation.rule == "artifact-hash-mismatch" and "known-signal evidence raw input" in violation.detail
        for violation in report.violations
    )


def test_catalog_match_ephemeris_cli_dispatches_candidate_local_output(tmp_path, capsys):
    candidate = _candidate(tmp_path)
    _write_candidate_ephemeris(candidate)

    result = main(["--root", str(tmp_path), "catalog", "match-ephemeris", candidate.candidate_id])

    assert result == 0
    assert "known_signal_ephemeris_match.json" in capsys.readouterr().out


def test_catalog_record_ephemeris_cli_writes_candidate_local_evidence(tmp_path, capsys):
    candidate = _candidate(tmp_path)
    raw_artifact = candidate.path / "literature" / "reviewed-row.txt"
    raw_artifact.write_text("reviewed source row", encoding="utf-8")

    result = main(
        [
            "--root", str(tmp_path), "catalog", "record-ephemeris", candidate.candidate_id,
            "--record-id", "literature-row-01",
            "--source-kind", "literature",
            "--source-name", "Reviewed source",
            "--source-uri", "https://example.invalid/source",
            "--raw-artifact", "literature/reviewed-row.txt",
            "--period-days", "3.0",
            "--epoch-bjd-tdb", "2457001.0",
            "--duration-hours", "2.0",
            "--retrieved-at", "2026-08-01T00:00:00Z",
            "--expires-at", "2026-09-01T00:00:00Z",
        ]
    )

    assert result == 0
    assert "decisions/known_signal_ephemerides.json" in capsys.readouterr().out


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
