"""Concurrent, resumable HTTP download engine with rich live progress.

This module is a target-neutral byte-transfer utility.  It does not read or
write candidate metadata, interpret sector or ephemeris values, or make any
astrophysical inference.  All target-specific decisions (which URLs to fetch,
where to place files) are made by the callers in ``ingest.py``.

Architecture
------------
``DownloadEngine`` accepts a list of ``DownloadItem`` records (URL + local
destination path + display label) and downloads them concurrently via
``concurrent.futures.ThreadPoolExecutor``.  Each file is streamed in 64 KB
chunks to a ``.tmp`` sibling so that a partially downloaded file never
replaces a complete one.  On success the temporary file is atomically renamed
to its destination.

Progress display
----------------
When *quiet* is ``False`` and ``sys.stdout`` is a TTY, a ``rich.progress``
live panel is shown with:
  * One overall task: ``N / M files completed``
  * One per-file sub-task: label, BarColumn, DownloadColumn,
    TransferSpeedColumn, TimeRemainingColumn

In quiet / non-TTY mode the progress object is replaced with a no-op context
manager and all feedback is written via the standard ``logging`` module.

Error handling
--------------
* **HTTP 429** — reads ``Retry-After`` header; falls back to exponential
  back-off (1 s, 2 s, 4 s …).  Up to *max_retries* total attempts.
* **HTTP 401 / 403** — raises ``DownloadAccessError`` immediately with a
  human-readable IP-restriction diagnosis.
* **Connection reset / timeout / WinError 10054** — resumes from the last
  committed byte offset using ``Range: bytes=X-`` on the next attempt.
  Up to *max_retries* total attempts.
* **Size mismatch** — if the server advertised a ``Content-Length`` and the
  completed file differs, the ``.tmp`` is removed and the download is retried
  from scratch.

Scientific boundary
-------------------
Downloading a file establishes acquisition provenance.  It does not assign
photometric quality, detect a transit signal, or constitute a validation
result.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 64 * 1024          # 64 KB per read
_DEFAULT_MAX_WORKERS = 4
_DEFAULT_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class DownloadItem:
    """One file to be downloaded.

    Attributes:
        url: Full HTTPS URL to retrieve.
        destination: Absolute local path for the completed file.
        label: Short human-readable label shown in the progress bar.
    """

    url: str
    destination: Path
    label: str


@dataclass
class DownloadResult:
    """Outcome of a completed single-file download.

    Attributes:
        destination: Absolute path of the completed local file.
        sha256: Hex-encoded SHA-256 digest of the file bytes as downloaded.
        source_uri: Original URL for the provenance record.
        bytes_downloaded: Total bytes transferred in this session (excludes
            any bytes already present when resuming).
        resumed: ``True`` when the download continued from a partial ``.tmp``.
    """

    destination: Path
    sha256: str
    source_uri: str
    bytes_downloaded: int
    resumed: bool = False


class DownloadAccessError(RuntimeError):
    """Raised when the server returns 401 or 403.

    Attributes:
        status_code: The HTTP status code received.
        url: The URL that triggered the access denial.
    """

    def __init__(self, status_code: int, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(
            "HTTP {code} Access Denied for {url!r}. "
            "This usually indicates an IP restriction, expired token, or "
            "MAST rate-ban.  Check your network/VPN settings or wait before retrying.".format(
                code=status_code, url=url
            )
        )


class DownloadError(RuntimeError):
    """Raised when a download fails after exhausting all retry attempts."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tmp_path(destination: Path) -> Path:
    """Return the ``.tmp`` sibling path used while a download is in progress."""
    return destination.with_suffix(destination.suffix + ".tmp")


def _is_tty() -> bool:
    """Return ``True`` when stdout is an interactive terminal."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ---------------------------------------------------------------------------
# DownloadEngine
# ---------------------------------------------------------------------------


class DownloadEngine:
    """Concurrent, resumable HTTP download engine.

    Args:
        max_workers: Maximum number of parallel download threads.
        chunk_size: Read buffer size in bytes (streamed to disk).
        max_retries: Maximum retry attempts per file (covers 429 and
            connection errors).
        quiet: When ``True``, suppress the ``rich.progress`` display and
            emit only ``logging`` messages.  Automatically forced when
            stdout is not a TTY.
    """

    def __init__(
        self,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        chunk_size: int = _CHUNK_SIZE,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        quiet: bool = False,
    ) -> None:
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.quiet = quiet or not _is_tty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_many(self, items: Sequence[DownloadItem]) -> List[DownloadResult]:
        """Download *items* concurrently and return one result per item.

        Args:
            items: Sequence of :class:`DownloadItem` records to fetch.

        Returns:
            List of :class:`DownloadResult` in the same order as *items*.

        Raises:
            DownloadAccessError: If any item returns HTTP 401 or 403.
            DownloadError: If any item fails after all retry attempts.
        """
        if not items:
            return []

        index_map = {id(item): idx for idx, item in enumerate(items)}

        if self.quiet:
            return self._download_many_quiet(list(items), index_map)
        return self._download_many_progress(list(items), index_map)

    # ------------------------------------------------------------------
    # Quiet (CI / non-TTY) path
    # ------------------------------------------------------------------

    def _download_many_quiet(
        self,
        items: List[DownloadItem],
        index_map: dict,
    ) -> List[DownloadResult]:
        results: List[Optional[DownloadResult]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_item = {
                pool.submit(self._download_one_quiet, item): item for item in items
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                idx = index_map[id(item)]
                results[idx] = future.result()
        return results  # type: ignore[return-value]

    def _download_one_quiet(self, item: DownloadItem) -> DownloadResult:
        logger.info("Downloading %s -> %s", item.label, item.destination)
        result = self._download_one(item, progress=None, task_id=None)
        logger.info(
            "Completed %s (%.1f MB, sha256=%s...)",
            item.label,
            result.bytes_downloaded / (1024 * 1024),
            result.sha256[:12],
        )
        return result

    # ------------------------------------------------------------------
    # Rich progress path
    # ------------------------------------------------------------------

    def _download_many_progress(
        self,
        items: List[DownloadItem],
        index_map: dict,
    ) -> List[DownloadResult]:
        from rich.logging import RichHandler
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        # Install RichHandler once so logger output appears above the progress bars.
        root_logger = logging.getLogger()
        if not any(isinstance(h, RichHandler) for h in root_logger.handlers):
            root_logger.addHandler(RichHandler(markup=True, rich_tracebacks=True))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            expand=True,
            transient=False,
        )

        total_task = progress.add_task(
            "[bold white]Overall", total=len(items), completed=0
        )

        results: List[Optional[DownloadResult]] = [None] * len(items)

        with progress:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_pair = {}
                for item in items:
                    task_id = progress.add_task(
                        "[green]{label}".format(label=item.label),
                        total=None,
                        start=True,
                    )
                    future = pool.submit(
                        self._download_one, item, progress, task_id
                    )
                    future_to_pair[future] = (item, task_id)

                for future in as_completed(future_to_pair):
                    item, task_id = future_to_pair[future]
                    idx = index_map[id(item)]
                    result = future.result()
                    results[idx] = result
                    progress.update(task_id, visible=False)
                    progress.advance(total_task)
                    progress.log(
                        "[green]OK[/green] {label} ({mb:.1f} MB)".format(
                            label=item.label,
                            mb=result.bytes_downloaded / (1024 * 1024),
                        )
                    )

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core single-file download (shared by both paths)
    # ------------------------------------------------------------------

    def _download_one(
        self,
        item: DownloadItem,
        progress: Optional[object],
        task_id: Optional[object],
    ) -> DownloadResult:
        """Download one file with retry/resume logic."""
        destination = Path(item.destination)
        tmp = _tmp_path(destination)
        url = item.url

        attempt = 0
        backoff = _INITIAL_BACKOFF_SECONDS
        total_bytes_this_session = 0
        resumed = False

        while attempt < self.max_retries:
            attempt += 1

            # Determine resume offset from existing .tmp
            resume_offset = tmp.stat().st_size if tmp.exists() else 0
            if resume_offset > 0:
                resumed = True

            headers: dict = {}
            if resume_offset > 0:
                headers["Range"] = "bytes={0}-".format(resume_offset)

            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(15, 60),
                ) as response:
                    status = response.status_code

                    # ---- access denied: fast fail ---------------------------
                    if status in (401, 403):
                        raise DownloadAccessError(status, url)

                    # ---- rate limited: back off and retry -------------------
                    if status == 429:
                        retry_after = response.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else backoff
                        msg = "[Rate Limit: {w:.0f}s bekleniyor...]".format(w=wait)
                        if progress is not None:
                            progress.log(  # type: ignore[union-attr]
                                "[yellow]{msg}[/yellow]".format(msg=msg)
                            )
                        else:
                            logger.warning(msg)
                        time.sleep(wait)
                        backoff = min(backoff * 2, 64.0)
                        continue

                    # ---- server error: retry with backoff -------------------
                    if status >= 500:
                        logger.warning(
                            "HTTP %d for %s (attempt %d/%d); retrying in %.0fs",
                            status,
                            item.label,
                            attempt,
                            self.max_retries,
                            backoff,
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 64.0)
                        continue

                    # ---- reject unexpected status codes ---------------------
                    if status not in (200, 206):
                        raise DownloadError(
                            "Unexpected HTTP {0} for {1!r}".format(status, url)
                        )

                    # If server ignores Range and returns 200, start fresh
                    if status == 200 and resume_offset > 0:
                        resume_offset = 0
                        resumed = False
                        tmp.unlink(missing_ok=True)

                    content_length_str = response.headers.get("Content-Length")
                    expected_total: Optional[int] = None
                    if content_length_str is not None:
                        try:
                            expected_total = int(content_length_str) + resume_offset
                        except ValueError:
                            expected_total = None

                    # Update rich task total once we know it
                    if progress is not None and expected_total is not None:
                        progress.update(  # type: ignore[union-attr]
                            task_id,
                            total=expected_total,
                            completed=resume_offset,
                        )

                    # ---- stream bytes to .tmp --------------------------------
                    write_mode = "ab" if resume_offset > 0 else "wb"
                    bytes_this_response = 0
                    digest = hashlib.sha256()

                    # Pre-hash bytes already in tmp when resuming
                    if resume_offset > 0 and tmp.exists():
                        with tmp.open("rb") as existing:
                            for pre_chunk in iter(
                                lambda: existing.read(8 * 1024 * 1024), b""
                            ):
                                digest.update(pre_chunk)

                    with tmp.open(write_mode) as fh:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            digest.update(chunk)
                            bytes_this_response += len(chunk)
                            total_bytes_this_session += len(chunk)
                            if progress is not None:
                                progress.advance(  # type: ignore[union-attr]
                                    task_id, len(chunk)
                                )

                    # ---- integrity check: size must match -------------------
                    actual_total = resume_offset + bytes_this_response
                    if expected_total is not None and actual_total != expected_total:
                        logger.warning(
                            "Size mismatch for %s: expected %d bytes, got %d; "
                            "retrying from scratch",
                            item.label,
                            expected_total,
                            actual_total,
                        )
                        tmp.unlink(missing_ok=True)
                        resume_offset = 0
                        resumed = False
                        backoff = min(backoff * 2, 64.0)
                        continue

                    # ---- atomic rename to final destination -----------------
                    tmp.replace(destination)
                    sha256_hex = digest.hexdigest()

                    return DownloadResult(
                        destination=destination,
                        sha256=sha256_hex,
                        source_uri=url,
                        bytes_downloaded=total_bytes_this_session,
                        resumed=resumed,
                    )

            except DownloadAccessError:
                raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OSError,
            ) as exc:
                # WinError 10054 and similar are OSError subclasses on Windows
                if attempt >= self.max_retries:
                    raise DownloadError(
                        "Download failed for {label!r} after {n} attempts: {exc}".format(
                            label=item.label, n=self.max_retries, exc=exc
                        )
                    ) from exc
                logger.warning(
                    "Connection error for %s (attempt %d/%d): %s; resuming in %.0fs",
                    item.label,
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                if progress is not None:
                    progress.log(  # type: ignore[union-attr]
                        "[yellow]! {label}: baglanti hatasi, {w:.0f}s sonra devam edilecek[/yellow]".format(
                            label=item.label, w=backoff
                        )
                    )
                time.sleep(backoff)
                backoff = min(backoff * 2, 64.0)

        raise DownloadError(
            "Download failed for {label!r} after {n} attempts".format(
                label=item.label, n=self.max_retries
            )
        )
