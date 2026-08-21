"""Candidate data ingestion: network fetch plus offline provenance recording.

The network fetchers retrieve selected SPOC or TESSCut light curves from MAST
via lightkurve. ``ingest_products`` is a pure function that copies downloaded
products into ``candidate/<id>/data/raw/`` and writes
``.provenance.json`` sidecars, satisfying the acquisition gate.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import quote

from .catalog import write_provenance_sidecar
from .workspace import CandidateWorkspace

Product = Tuple[Path, str]  # (local product path, source URI)
_PROVIDERS = ("spoc",)


def _validate_provider(provider: str) -> str:
    if provider not in _PROVIDERS:
        raise ValueError(
            "unsupported TESS data provider {0!r}; choose 'spoc'".format(provider)
        )
    return provider


def _sector_value(row: object) -> Optional[int]:
    try:
        return int(row["sequence_number"]) or None  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


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


def _download_spoc_product(row: object, staging: Path) -> Path:
    """Download archive bytes directly instead of serializing a Lightkurve object."""
    try:
        from astroquery.mast import Observations
    except ImportError as exc:  # pragma: no cover - declared discovery dependency
        raise RuntimeError("astroquery is required for SPOC product ingestion") from exc

    destination = staging / _mast_product_filename(row)
    status, message, _ = Observations.download_file(
        _mast_product_data_uri(row),
        local_path=str(destination),
        cache=False,
        verbose=False,
    )
    if status != "COMPLETE" or not destination.is_file():
        raise RuntimeError("MAST product download failed: {0}".format(message or status))
    return destination


def ingest_products(
    workspace: CandidateWorkspace,
    products: Sequence[Product],
    fetched_by: str = "exonym-ingest/1.2.0",
) -> List[Path]:
    """Copy products into ``data/raw/`` and write provenance sidecars.

    Raises ``FileExistsError`` if a product name already exists in the raw
    directory (no-clobber rule).
    """
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

    staging = Path(tempfile.mkdtemp(prefix=".ingest-staging-", dir=str(raw)))
    committed: List[Path] = []
    try:
        staged_pairs = []
        for product, source_uri, destination, sidecar in planned:
            staged_product = staging / destination.name
            shutil.copy2(product, staged_product)
            staged_sidecar = write_provenance_sidecar(
                staged_product, source_uri, fetched_by=fetched_by
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


def fetch_tess_products(
    workspace: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    exptime: int = 120,
    provider: str = "spoc",
) -> List[Product]:
    """Download light curves from the selected MAST provider.

    Returns ``(local_path, source_uri)`` pairs staged in a temporary
    directory. ``provider`` is currently limited to official SPOC products,
    whose archived bytes and MAST data URI can be retained together. The caller
    passes products to ``ingest_products``.
    """
    provider = _validate_provider(provider)
    try:
        import lightkurve as lk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightkurve is required for ingestion") from exc

    tic = workspace.metadata["identifiers"].get("tic")
    if not tic:
        raise ValueError("a TIC identifier is required for TESS ingestion")
    target = "TIC {0}".format(tic)

    search = lk.search_lightcurve(target, author="SPOC", exptime=exptime)
    if not search:
        return []

    products: List[Product] = []
    staging = Path(tempfile.mkdtemp(prefix="exonym-ingest-"))
    for index in range(len(search)):
        row = search.table[index]
        sector_value = _sector_value(row)
        if sectors is not None and sector_value is not None and sector_value not in sectors:
            continue

        fits_path = _download_spoc_product(row, staging)
        products.append((fits_path, _mast_product_uri(row)))
    return products


def fetch_tess_tpfs(
    workspace: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    exptime: int = 120,
    provider: str = "spoc",
) -> List[Product]:
    """Download SPOC target pixel files from MAST (network access required).

    TPF products are staged as ``s{sec:04d}_tp.fits`` so the acquisition gate's
    provenance sidecar convention (``<stem>.provenance.json``) applies to them
    exactly like light curves — the ``tp`` stem marker is also what
    ``inputs.load_tpf_cubes`` uses to distinguish TPFs from light curves.
    """
    provider = _validate_provider(provider)
    try:
        import lightkurve as lk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightkurve is required for ingestion") from exc

    tic = workspace.metadata["identifiers"].get("tic")
    if not tic:
        raise ValueError("a TIC identifier is required for TESS ingestion")
    target = "TIC {0}".format(tic)

    search = lk.search_targetpixelfile(target, author="SPOC", exptime=exptime)
    if not search:
        return []

    products: List[Product] = []
    staging = Path(tempfile.mkdtemp(prefix="exonym-ingest-"))
    for index in range(len(search)):
        row = search.table[index]
        sector_value = _sector_value(row)
        if sectors is not None and sector_value is not None and sector_value not in sectors:
            continue

        fits_path = _download_spoc_product(row, staging)
        products.append((fits_path, _mast_product_uri(row)))
    return products
