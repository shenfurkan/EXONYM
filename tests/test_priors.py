"""Tests for catalog-prior ingestion without external network access."""

import json

from exonym.priors import fetch_exofop_priors
from exonym.workspace import create_candidate


class _FakeResponse:
    status = 200

    def __init__(self, body):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_fetch_exofop_priors_filters_tic_and_writes_signal_configs(tmp_path, monkeypatch):
    """Only matching TIC rows become normalized per-signal transit configs."""
    workspace = create_candidate(tmp_path, "catalog-prior-test", tic="123456789")
    csv_body = "\n".join(
        (
            "TOI,TIC ID,Period (days),Epoch (BJD),Depth (ppm),Duration (hours)",
            "100.01,123456789,4.12345678,2458123.45678,321.987,2.345",
            "100.02,123456789,8.76543219,2458130.12567,654.321,4.567",
            "101.01,987654321,3.5,2458000.0,999.0,1.5",
        )
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append({"url": request.full_url, "timeout": timeout})
        return _FakeResponse(csv_body)

    monkeypatch.setattr("exonym.priors.urllib.request.urlopen", fake_urlopen)

    written = fetch_exofop_priors(workspace)

    assert [path.name for path in written] == [
        "transit_config.01.json",
        "transit_config.02.json",
    ]
    assert calls == [
        {
            "url": "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv",
            "timeout": 30,
        }
    ]

    first = json.loads(written[0].read_text(encoding="utf-8"))
    second = json.loads(written[1].read_text(encoding="utf-8"))
    assert first == {
        "transit": {
            "period_days": 4.123457,
            "epoch_btjd": 1123.45678,
            "depth_ppm": 321.99,
            "duration_hours": 2.35,
            "source": "nasa-exofop",
        }
    }
    assert second == {
        "transit": {
            "period_days": 8.765432,
            "epoch_btjd": 1130.12567,
            "depth_ppm": 654.32,
            "duration_hours": 4.57,
            "source": "nasa-exofop",
        }
    }
