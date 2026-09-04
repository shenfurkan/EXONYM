"""Candidate data ingestion: network retrieval and offline provenance records.

Network fetchers retrieve selected official products from MAST through optional
dependencies. The ingestion entry point copies caller-provided downloaded
files into candidate data/raw and writes provenance sidecars, so later
workflow steps can bind raw bytes to their recorded source URI.

Scientific Boundary:
    Downloading or recording a product establishes acquisition provenance, not
    photometric quality, a detected signal, or an astrophysical conclusion.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

from .catalog import write_provenance_sidecar
from .download import DownloadEngine, DownloadItem
from .workspace import CandidateWorkspace

Product = Tuple[Path, str]  # (local product path, source URI)
_PROVIDERS = ("spoc",)


class FetchedProducts(list):
    """List-like downloaded products that own one temporary staging directory.

    Use a fetched batch as a context manager when combining products from
    multiple archive queries. Passing one batch directly to :func:`ingest_products`
    also releases its staging directory after the ingest attempt, whether or
    not the candidate-local commit succeeds.

    Attributes:
        _sha256_cache: Optional mapping from staged product path to its
            pre-computed hex SHA-256 digest.  Populated by the download engine
            to avoid re-reading FITS files during provenance sidecar creation.
    """

    def __init__(self, staging_path: Optional[Path] = None) -> None:
        super().__init__()
        self._staging_path = staging_path
        self._cleaned = False
        self._sha256_cache: dict = {}

    @property
    def staging_path(self) -> Optional[Path]:
        """Return the owned temporary directory, if this batch downloaded files."""
        return self._staging_path

    def cleanup(self) -> None:
        """Remove the owned staging directory once product bytes are no longer needed."""
        if self._staging_path is None or self._cleaned:
            return
        self._cleaned = True
        shutil.rmtree(self._staging_path, ignore_errors=True)

    def __enter__(self) -> "FetchedProducts":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.cleanup()
        return False


def _validate_provider(provider: str) -> str:
    if provider not in _PROVIDERS:
        raise ValueError(
            "unsupported TESS data provider {0!r}; choose 'spoc'".format(provider)
        )
    return provider


def _coerce_sector_value(value: object) -> Optional[int]:
    """Parse one positive sector value without guessing from product filenames."""
    try:
        sector = int(value)
    except (TypeError, ValueError, OverflowError):
        text = str(value).strip()
        if text.lower().startswith("s"):
            text = text[1:]
        try:
            sector = int(text)
        except (TypeError, ValueError, OverflowError):
            return None
    return sector if sector > 0 else None


def _sector_value(row: object) -> Optional[int]:
    try:
        value = row["sequence_number"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return None
    return _coerce_sector_value(value)


def _requested_sectors(sectors: Optional[Sequence[int]]) -> Optional[Set[int]]:
    """Normalize an explicit sector filter or reject malformed selections loudly."""
    if sectors is None:
        return None
    normalized = {_coerce_sector_value(sector) for sector in sectors}
    if None in normalized:
        raise ValueError("sectors must contain positive integer values")
    return {int(sector) for sector in normalized}


def _mast_product_data_uri(row: object) -> str:
    """Return the MAST data URI recorded by the product search result."""
    columns = getattr(row, "colnames", ())
    if "dataURI" in columns:
        data_uri = str(row["dataURI"]).strip()
        if data_uri.startswith("mast:"):
            return data_uri
    filename = str(row["productFilename"]) if "productFilename" in columns else str(row["obs_id"])
    filename = Path(filename).name
    if not filename:
        raise ValueError("MAST product row lacks a usable dataURI or product filename")
    return "mast:TESS/product/{0}".format(filename)


def _mast_product_filename(row: object) -> str:
    columns = getattr(row, "colnames", ())
    value = row["productFilename"] if "productFilename" in columns else row["obs_id"]
    filename = Path(str(value)).name
    if not filename:
        raise ValueError("MAST product row lacks a usable product filename")
    return filename


def _mast_product_uri(row: object) -> str:
    data_uri = _mast_product_data_uri(row)
    return "https://mast.stsci.edu/api/v0.1/Download/file?uri={0}".format(
        quote(data_uri, safe=":/")
    )


def _fetch_spoc_products(
    search: object,
    sectors: Optional[Sequence[int]],
    quiet: bool = False,
    workers: int = 4,
) -> "FetchedProducts":
    """Download selected archive rows into an owned temporary staging directory.

    An explicit sector selection is a provenance constraint. Rows without a
    usable ``sequence_number`` are therefore excluded rather than downloaded
    under an unknown sector. A failed download removes all partially staged
    bytes before propagating the archive error.

    Args:
        search: Lightkurve search result whose product table is iterated.
        sectors: Optional sector filter; ``None`` downloads all rows.
        quiet: Suppress the ``rich.progress`` display (CI / non-TTY mode).
        workers: Maximum number of concurrent download threads.

    Returns:
        :class:`FetchedProducts` owning a temporary staging directory, or an
        empty batch when the sector filter excludes all rows.
    """
    requested_sectors = _requested_sectors(sectors)
    selected_rows: List[object] = []
    for index in range(len(search)):  # type: ignore[arg-type]
        row = search.table[index]  # type: ignore[union-attr]
        sector_value = _sector_value(row)
        if requested_sectors is not None:
            if sector_value is None or sector_value not in requested_sectors:
                continue
        selected_rows.append(row)

    if not selected_rows:
        return FetchedProducts()

    staging = Path(tempfile.mkdtemp(prefix="exonym-ingest-"))
    products = FetchedProducts(staging)
    completed = False
    try:
        items = [
            DownloadItem(
                url=_mast_product_uri(row),
                destination=staging / _mast_product_filename(row),
                label=_mast_product_filename(row),
            )
            for row in selected_rows
        ]
        engine = DownloadEngine(max_workers=workers, quiet=quiet)
        results = engine.download_many(items)
        for result in results:
            products.append((result.destination, result.source_uri))
            # Cache the pre-computed SHA-256 so ingest_products can pass it
            # to write_provenance_sidecar without re-reading the FITS file.
            products._sha256_cache[result.destination] = result.sha256
        completed = True
        return products
    finally:
        if not completed:
            products.cleanup()


def ingest_products(
    workspace: CandidateWorkspace,
    products: Sequence[Product],
    fetched_by: str = "exonym-ingest/1.2.0",
) -> List[Path]:
    """Atomically ingest downloaded products and write provenance sidecars.

    The operation plans every destination before modifying data/raw, stages
    files and sidecars together, and removes committed files if a later move
    fails. Existing raw names and sidecars are never overwritten.

    Args:
        workspace: Candidate workspace that owns the raw-product directory.
        products: Local downloaded path and source-URI pairs to ingest.
        fetched_by: Retrieval-agent label recorded in each provenance sidecar.

    Returns:
        Destination paths for successfully ingested raw products in input
        order, or an empty list when products is empty.

    Raises:
        FileExistsError: If a destination product or provenance sidecar already
            exists.
        ValueError: If multiple batch entries would collide after naming.
        OSError: If copying, sidecar creation, or final moves cannot complete.
    """
    fetched_products = products if isinstance(products, FetchedProducts) else None
    try:
        raw = workspace.path / "data" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        planned = []
        seen_destinations = set()
        for product, source_uri in products:
            destination = raw / Path(product).name
            sidecar = destination.with_name(destination.stem + ".provenance.json")
            if destination.exists() or sidecar.exists():
                raise FileExistsError("raw product or provenance sidecar already exists: {0}".format(destination))
            if destination.name in seen_destinations or sidecar.name in seen_destinations:
                raise ValueError("ingest batch contains colliding product or sidecar names")
            seen_destinations.update((destination.name, sidecar.name))
            planned.append((Path(product), str(source_uri), destination, sidecar))
        if not planned:
            return []

        sha256_cache: dict = getattr(products, "_sha256_cache", {}) if fetched_products is not None else {}
        staging = Path(tempfile.mkdtemp(prefix=".ingest-staging-", dir=str(raw)))
        committed: List[Path] = []
        try:
            staged_pairs = []
            for product, source_uri, destination, sidecar in planned:
                staged_product = staging / destination.name
                shutil.copy2(product, staged_product)
                # Use the pre-computed SHA-256 from the download engine when
                # available to avoid re-reading large FITS files from disk.
                precomputed_sha256 = sha256_cache.get(Path(product))
                staged_sidecar = write_provenance_sidecar(
                    staged_product, source_uri, fetched_by=fetched_by,
                    sha256=precomputed_sha256,
                )
                staged_pairs.append((staged_product, staged_sidecar, destination, sidecar))
            for staged_product, staged_sidecar, destination, sidecar in staged_pairs:
                shutil.move(str(staged_product), str(destination))
                committed.append(destination)
                try:
                    shutil.move(str(staged_sidecar), str(sidecar))
                    committed.append(sidecar)
                except OSError:
                    destination.unlink(missing_ok=True)
                    committed.pop()
                    raise
            return [destination for _, _, destination, _ in planned]
        except Exception:
            for path in reversed(committed):
                path.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        if fetched_products is not None:
            fetched_products.cleanup()


def fetch_tess_products(
    workspace: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    exptime: Optional[int] = None,
    provider: str = "spoc",
    quiet: bool = False,
    workers: int = 4,
) -> FetchedProducts:
    """Fetch official mission light curves into caller-owned staging paths.

    The returned list owns a temporary staging directory. Use it as a context
    manager when combining batches, or pass it directly to ``ingest_products``;
    either path removes the staged bytes after consumption. No candidate data
    is modified until that separate ingestion step succeeds.

    Args:
        workspace: Candidate workspace that supplies the mission identifier.
        sectors: Optional archive-sector selection applied before download.
        exptime: Optional archive-product cadence filter in seconds. This
            selection filter is not used as a scientific integration time;
            downstream analyses derive exposure from raw FITS metadata.
        provider: Supported official archive provider label.
        quiet: Suppress the ``rich.progress`` display (CI / non-TTY mode).
        workers: Maximum number of concurrent download threads.

    Returns:
        Context-manageable local staging-path and source-URI pairs for
        downloaded light curves.

    Raises:
        ValueError: If provider or required candidate mission identifier is
            invalid.
        RuntimeError: If optional archive dependencies or product download are
            unavailable.
    """
    _validate_provider(provider)
    try:
        import lightkurve as lk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightkurve is required for ingestion") from exc

    tic = workspace.metadata["identifiers"].get("tic")
    if not tic:
        raise ValueError("a TIC identifier is required for TESS ingestion")
    target = "TIC {0}".format(tic)

    search_kwargs = {"author": "SPOC"}
    if exptime is not None:
        search_kwargs["exptime"] = exptime
    search = lk.search_lightcurve(target, **search_kwargs)
    if not search:
        return FetchedProducts()

    return _fetch_spoc_products(search, sectors, quiet=quiet, workers=workers)


def fetch_tess_tpfs(
    workspace: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    exptime: Optional[int] = None,
    provider: str = "spoc",
    quiet: bool = False,
    workers: int = 4,
) -> FetchedProducts:
    """Fetch official target-pixel files into caller-owned staging paths.

    Target-pixel products use the established filename marker that lets the
    input loader distinguish them from light curves after provenance-aware
    ingestion. The returned list owns its temporary staging directory; use it
    as a context manager when combining batches, or pass it directly to
    ``ingest_products``. Network retrieval does not write candidate data until
    that separate ingestion step succeeds.

    Args:
        workspace: Candidate workspace that supplies the mission identifier.
        sectors: Optional archive-sector selection applied before download.
        exptime: Optional archive-product cadence filter in seconds. This
            selection filter is not used as a scientific integration time;
            downstream analyses derive exposure from raw FITS metadata.
        provider: Supported official archive provider label.
        quiet: Suppress the ``rich.progress`` display (CI / non-TTY mode).
        workers: Maximum number of concurrent download threads.

    Returns:
        Context-manageable local staging-path and source-URI pairs for
        downloaded target-pixel products.

    Raises:
        ValueError: If provider or required candidate mission identifier is
            invalid.
        RuntimeError: If optional archive dependencies or product download are
            unavailable.
    """
    _validate_provider(provider)
    try:
        import lightkurve as lk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightkurve is required for ingestion") from exc

    tic = workspace.metadata["identifiers"].get("tic")
    if not tic:
        raise ValueError("a TIC identifier is required for TESS ingestion")
    target = "TIC {0}".format(tic)

    search_kwargs = {"author": "SPOC"}
    if exptime is not None:
        search_kwargs["exptime"] = exptime
    search = lk.search_targetpixelfile(target, **search_kwargs)
    if not search:
        return FetchedProducts()

    # SCIENTIFIC_BOUNDARY: Retain the MAST URI with each downloaded byte stream
    # so later ingestion can record provenance without assigning data quality.
    return _fetch_spoc_products(search, sectors, quiet=quiet, workers=workers)
