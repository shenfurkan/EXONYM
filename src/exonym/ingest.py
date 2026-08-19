"""Candidate data ingestion: network fetch plus offline provenance recording.

The network fetchers retrieve selected SPOC or TESSCut light curves from MAST
via lightkurve. ``ingest_products`` is a pure function that copies downloaded
products into ``candidate/<id>/data/raw/`` and writes
``.provenance.json`` sidecars, satisfying the acquisition gate.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .catalog import write_provenance_sidecar
from .workspace import CandidateWorkspace

Product = Tuple[Path, str]  # (local product path, source URI)
_PROVIDERS = ("spoc", "tesscut")


def _validate_provider(provider: str) -> str:
    if provider not in _PROVIDERS:
        raise ValueError(
            "unsupported TESS data provider {0!r}; choose 'spoc' or 'tesscut'".format(provider)
        )
    return provider


def _sector_value(row: object) -> Optional[int]:
    try:
        return int(row["sequence_number"]) or None  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


def _mast_product_uri(row: object) -> str:
    columns = getattr(row, "colnames", ())
    obs_id = str(row["obs_id"]) if "obs_id" in columns else str(row["productFilename"])
    return "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{0}".format(
        obs_id
    )


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
            os.link(staged_product, destination)
            committed.append(destination)
            try:
                os.link(staged_sidecar, sidecar)
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
    directory. ``provider`` must be ``"spoc"`` or ``"tesscut"``; exposure
    time selection applies only to SPOC. The caller passes products to
    ``ingest_products``.
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

    products: List[Product] = []
    if provider == "tesscut":
        staging = Path(tempfile.mkdtemp(prefix="exonym-ingest-"))
        requested_sectors = tuple(dict.fromkeys(sectors)) if sectors is not None else (None,)
        for requested_sector in requested_sectors:
            search = lk.search_tesscut(target, sector=requested_sector)
            if not search:
                continue
            for index in range(len(search)):
                row = search.table[index]
                sector_value = _sector_value(row)
                if sectors is not None and sector_value is not None and sector_value not in sectors:
                    continue

                tpf = search[index].download()
                light_curve = tpf.to_lightcurve()
                filename_sector = sector_value if sector_value is not None else requested_sector
                fits_path = staging / "s{0:04d}_lc.fits".format(filename_sector or index)
                light_curve.to_fits(path=fits_path, overwrite=True)
                products.append((fits_path, _mast_product_uri(row)))
        return products

    search = lk.search_lightcurve(target, author="SPOC", exptime=exptime)
    if not search:
        return []

    staging = Path(tempfile.mkdtemp(prefix="exonym-ingest-"))
    for index in range(len(search)):
        row = search.table[index]
        sector_value = _sector_value(row)
        if sectors is not None and sector_value is not None and sector_value not in sectors:
            continue

        light_curve = search[index].download()
        fits_path = staging / "s{0:04d}_lc.fits".format(sector_value or index)
        light_curve.to_fits(path=fits_path, overwrite=True)
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
    if provider == "tesscut":
        raise ValueError("TESSCut supports light-curve ingestion only; request TPFs from 'spoc'")
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

        tpf = search[index].download()
        fits_path = staging / "s{0:04d}_tp.fits".format(sector_value or index)
        tpf.to_fits(str(fits_path), overwrite=True)
        products.append((fits_path, _mast_product_uri(row)))
    return products
