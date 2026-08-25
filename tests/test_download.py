"""Tests for the exonym.download module.

All tests use synthetic data and unittest.mock; no network calls are made.
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from exonym.download import (
    DownloadAccessError,
    DownloadEngine,
    DownloadError,
    DownloadItem,
    DownloadResult,
    _tmp_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CONTENT = b"FITS" * (64 * 1024 // 4 * 3 + 17)  # ~192 KB of fake bytes


def _make_item(tmp_path: Path, name: str = "test.fits") -> DownloadItem:
    return DownloadItem(
        url="https://mast.stsci.edu/download/{0}".format(name),
        destination=tmp_path / name,
        label=name,
    )


def _mock_response(
    content: bytes,
    status_code: int = 200,
    headers: dict | None = None,
) -> MagicMock:
    """Build a mock requests.Response that streams *content* in 64 KB chunks."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {"Content-Length": str(len(content))}

    def _iter_content(chunk_size: int = 65536) -> Iterator[bytes]:
        offset = 0
        while offset < len(content):
            yield content[offset : offset + chunk_size]
            offset += chunk_size

    response.iter_content = _iter_content
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


# ---------------------------------------------------------------------------
# Test 1: Happy-path single download
# ---------------------------------------------------------------------------


def test_download_success(tmp_path: Path) -> None:
    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=1)

    with patch("requests.get", return_value=_mock_response(_FAKE_CONTENT)):
        results = engine.download_many([item])

    assert len(results) == 1
    result = results[0]
    assert result.destination == item.destination
    assert result.destination.is_file()
    assert result.bytes_downloaded == len(_FAKE_CONTENT)
    assert not result.resumed

    # Verify the SHA-256 is correct
    digest = hashlib.sha256(_FAKE_CONTENT).hexdigest()
    assert result.sha256 == digest

    # No .tmp left behind
    assert not _tmp_path(item.destination).exists()


# ---------------------------------------------------------------------------
# Test 2: Resume from partial .tmp
# ---------------------------------------------------------------------------


def test_download_resume(tmp_path: Path) -> None:
    item = _make_item(tmp_path)
    partial = _FAKE_CONTENT[: len(_FAKE_CONTENT) // 2]
    remaining = _FAKE_CONTENT[len(partial) :]

    # Pre-write partial .tmp to simulate an interrupted download
    tmp = _tmp_path(item.destination)
    tmp.write_bytes(partial)

    engine = DownloadEngine(quiet=True, max_retries=2)
    captured_headers: list[dict] = []

    def fake_get(url: str, headers: dict, **kwargs: object) -> MagicMock:
        captured_headers.append(dict(headers))
        # Return 206 for range request
        resp = _mock_response(
            remaining,
            status_code=206,
            headers={"Content-Length": str(len(remaining))},
        )
        return resp

    with patch("requests.get", side_effect=fake_get):
        results = engine.download_many([item])

    result = results[0]
    assert result.resumed
    # Range header must be set
    assert any("Range" in h for h in captured_headers)
    range_header = next(h["Range"] for h in captured_headers if "Range" in h)
    assert range_header == "bytes={0}-".format(len(partial))
    # Full file written
    assert result.destination.read_bytes() == _FAKE_CONTENT


# ---------------------------------------------------------------------------
# Test 3: HTTP 429 with Retry-After header
# ---------------------------------------------------------------------------


def test_download_rate_limit_retry_after(tmp_path: Path) -> None:
    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=3)
    call_count = 0

    def fake_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response(b"", status_code=429, headers={"Retry-After": "0"})
        return _mock_response(_FAKE_CONTENT)

    with patch("requests.get", side_effect=fake_get), patch("time.sleep") as mock_sleep:
        results = engine.download_many([item])

    assert results[0].destination.is_file()
    assert call_count == 2
    mock_sleep.assert_called_once_with(0.0)


# ---------------------------------------------------------------------------
# Test 4: HTTP 429 exponential back-off (no Retry-After)
# ---------------------------------------------------------------------------


def test_download_rate_limit_exp_backoff(tmp_path: Path) -> None:
    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=4)
    call_count = 0

    def fake_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return _mock_response(b"", status_code=429, headers={})
        return _mock_response(_FAKE_CONTENT)

    sleep_calls: list[float] = []
    with patch("requests.get", side_effect=fake_get), patch(
        "time.sleep", side_effect=lambda w: sleep_calls.append(w)
    ):
        results = engine.download_many([item])

    assert results[0].destination.is_file()
    # First back-off 1 s, second 2 s
    assert sleep_calls[0] == 1.0
    assert sleep_calls[1] == 2.0


# ---------------------------------------------------------------------------
# Test 5: HTTP 403 fast-fail
# ---------------------------------------------------------------------------


def test_download_403_fast_fail(tmp_path: Path) -> None:
    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=5)

    with patch("requests.get", return_value=_mock_response(b"", status_code=403)), pytest.raises(
        DownloadAccessError
    ) as exc_info:
        engine.download_many([item])

    assert exc_info.value.status_code == 403
    assert item.url in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 6: Connection reset retried then raises
# ---------------------------------------------------------------------------


def test_download_connection_reset_exhausts_retries(tmp_path: Path) -> None:
    import requests as req_module

    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=3)

    with patch(
        "requests.get",
        side_effect=req_module.exceptions.ConnectionError("WinError 10054"),
    ), patch("time.sleep"), pytest.raises(DownloadError):
        engine.download_many([item])


# ---------------------------------------------------------------------------
# Test 7: Connection reset succeeds on retry
# ---------------------------------------------------------------------------


def test_download_connection_reset_retries_successfully(tmp_path: Path) -> None:
    import requests as req_module

    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=3)
    call_count = 0

    def fake_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise req_module.exceptions.ConnectionError("reset")
        return _mock_response(_FAKE_CONTENT)

    with patch("requests.get", side_effect=fake_get), patch("time.sleep"):
        results = engine.download_many([item])

    assert results[0].destination.is_file()
    assert call_count == 2


# ---------------------------------------------------------------------------
# Test 8: Quiet mode produces no rich import side-effects on success
# ---------------------------------------------------------------------------


def test_download_quiet_no_rich_import(tmp_path: Path) -> None:
    """quiet=True must succeed without importing rich.progress internals."""
    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=1)

    import sys
    rich_progress_before = "rich.progress" in sys.modules

    with patch("requests.get", return_value=_mock_response(_FAKE_CONTENT)):
        results = engine.download_many([item])

    # If rich.progress was not already imported, quiet mode must not have imported it.
    if not rich_progress_before:
        assert "rich.progress" not in sys.modules
    assert results[0].destination.is_file()


# ---------------------------------------------------------------------------
# Test 9: Concurrent downloads — no race on staging directory
# ---------------------------------------------------------------------------


def test_download_concurrent_no_race(tmp_path: Path) -> None:
    names = ["file_{0}.fits".format(i) for i in range(6)]
    items = [_make_item(tmp_path, name) for name in names]
    engine = DownloadEngine(quiet=True, max_workers=4, max_retries=1)

    # Each response has distinct content so we can verify individuality
    contents = {name: (name.encode() * 1000) for name in names}
    call_lock = threading.Lock()

    def fake_get(url: str, **kwargs: object) -> MagicMock:
        name = url.split("/")[-1]
        with call_lock:
            content = contents[name]
        time.sleep(0.01)  # simulate slight I/O latency
        return _mock_response(content)

    with patch("requests.get", side_effect=fake_get):
        results = engine.download_many(items)

    assert len(results) == len(items)
    for result, item, name in zip(results, items, names):
        assert result.destination == item.destination
        assert result.destination.read_bytes() == contents[name]


# ---------------------------------------------------------------------------
# Test 10: Pre-computed SHA-256 is recorded in DownloadResult
# ---------------------------------------------------------------------------


def test_download_sha256_matches_content(tmp_path: Path) -> None:
    item = _make_item(tmp_path)
    engine = DownloadEngine(quiet=True, max_retries=1)

    with patch("requests.get", return_value=_mock_response(_FAKE_CONTENT)):
        results = engine.download_many([item])

    result = results[0]
    expected_sha256 = hashlib.sha256(_FAKE_CONTENT).hexdigest()
    assert result.sha256 == expected_sha256


# ---------------------------------------------------------------------------
# Test 11: make_provenance reuses pre-computed sha256 (no file re-read)
# ---------------------------------------------------------------------------


def test_make_provenance_uses_precomputed_sha256(tmp_path: Path) -> None:
    from exonym.catalog import make_provenance

    product = tmp_path / "product.fits"
    product.write_bytes(b"REAL CONTENT")

    precomputed = "aabbccddeeff" * 4  # fake 48-char hex (not real SHA-256 length but enough to test)

    with patch("exonym.catalog._sha256") as mock_sha256:
        record = make_provenance(product, "https://example.com/file", sha256=precomputed)

    # _sha256 must NOT be called when sha256 is supplied
    mock_sha256.assert_not_called()
    assert record["sha256"] == precomputed


# ---------------------------------------------------------------------------
# Test 12: write_provenance_sidecar propagates sha256 kwarg
# ---------------------------------------------------------------------------


def test_write_provenance_sidecar_uses_precomputed_sha256(tmp_path: Path) -> None:
    from exonym.catalog import write_provenance_sidecar
    import json

    product = tmp_path / "product.fits"
    product.write_bytes(b"FITS DATA")
    precomputed = hashlib.sha256(b"FITS DATA").hexdigest()

    sidecar = write_provenance_sidecar(product, "https://example.com/file", sha256=precomputed)

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["sha256"] == precomputed
