"""Universal TRILEGAL stellar population downloader for TREX.

Provides a certifi-backed HTTPS downloader with exponential backoff,
cross-platform CA bundle resolution, and candidate-local caching.

References
----------
* Girardi et al. (2005), ADS ``2005A&A...436..895G``, DOI
  ``10.1051/0004-6361:20042352`` – TRILEGAL.

Coordinates are ICRS degrees; field radius is degrees; limiting magnitude is
mag; HTTP timeouts/backoff are seconds; and output is an opaque candidate-local
CSV population realization.  Network failure returns no population file and
must be handled as unavailable scene evidence, not as zero background stars or
a claim-eligible result.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import certifi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from html.parser import HTMLParser

_TRILEGAL_URL_V16 = "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6"
_TRILEGAL_URL_V15 = "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.5"
_MAX_RETRIES = 25
_BACKOFF_FACTOR = 2.0
_SOCKET_TIMEOUT = 20.0


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.data_link: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "a" and self.data_link is None:
            for name, value in attrs:
                if name == "href" and "/tmp/" in value:
                    self.data_link = value
                    break


def _create_session() -> requests.Session:
    s = requests.Session()
    s.verify = certifi.where()
    retry = Retry(
        total=_MAX_RETRIES, backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def fetch_trilegal(
    ra_deg: float, dec_deg: float, target_id: int,
    cache_dir: Path, field_radius_deg: float = 0.1,
    mag_limit: float = 21.0, verbose: bool = False,
) -> Optional[Path]:
    """Download TRILEGAL stellar population, caching to disk.

    Returns Path to CSV, or None if unreachable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{target_id}_TRILEGAL.csv"
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path
    for version, url in [("1.6", _TRILEGAL_URL_V16), ("1.5", _TRILEGAL_URL_V15)]:
        try:
            result = _query_trilegal(
                url, ra_deg, dec_deg, cache_path,
                field_radius_deg, mag_limit, version == "1.6", verbose,
            )
            if result is not None:
                return result
        except Exception as e:
            if verbose:
                print(f"TREX TRILEGAL v{version}: {e}")
    return None


def _query_trilegal(
    url: str, ra_deg: float, dec_deg: float, cache_path: Path,
    field_deg: float, mag_lim: float, tess_photsys: bool, verbose: bool,
) -> Optional[Path]:
    """Submit a TRILEGAL form and download results."""
    s = _create_session()
    photsys = (
        "tab_mag_odfnew/tab_mag_TESS_2mass.dat"
        if tess_photsys else "tab_mag_odfnew/tab_mag_2mass.dat"
    )
    s.get(url, timeout=_SOCKET_TIMEOUT)
    data = {
        "gal_coord": "2", "eq_alpha": str(ra_deg),
        "eq_delta": str(dec_deg), "field": str(field_deg),
        "photsys_file": photsys, "icm_lim": "1",
        "mag_lim": str(mag_lim), "binary_kind": "0",
    }
    resp = s.post(url, data=data, timeout=_SOCKET_TIMEOUT)
    resp.raise_for_status()
    parser = _LinkParser()
    parser.feed(resp.text)
    if parser.data_link is None:
        return None
    time.sleep(5)
    data_url = f"https://stev.oapd.inaf.it{parser.data_link[3:]}"
    for _ in range(_MAX_RETRIES):
        try:
            rd = s.get(data_url, timeout=_SOCKET_TIMEOUT)
            rd.raise_for_status()
            if "#TRILEGAL normally terminated" in rd.text:
                cache_path.write_text(rd.text, encoding="utf-8")
                return cache_path
        except Exception:
            pass
        time.sleep(10)
    return None


__all__ = ["fetch_trilegal"]