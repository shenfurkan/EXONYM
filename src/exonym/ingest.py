"""Candidate data ingestion: network fetch plus offline provenance recording.

The network fetcher (``fetch_tess_products``) downloads SPOC products from
MAST via lightkurve. ``ingest_products`` is a pure function that copies the
downloaded products into ``candidate/<id>/data/raw/`` and writes
``.provenance.json`` sidecars, satisfying the acquisition gate.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .catalog import write_provenance_sidecar
from .workspace import CandidateWorkspace

Product = Tuple[Path, str]  # (local product path, source URI)


def ingest_products(
    workspace: CandidateWorkspace,
    products: Sequence[Product],
    fetched_by: str = "exonym-ingest/1.0.0",
) -> List[Path]:
    """Copy products into ``data/raw/`` and write provenance sidecars.

    Raises ``FileExistsError`` if a product name already exists in the raw
    directory (no-clobber rule).
    """
    raw = workspace.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for product, source_uri in products:
        destination = raw / Path(product).name
        if destination.exists():
            raise FileExistsError("raw product already exists: {0}".format(destination))
        shutil.copy2(product, destination)
        write_provenance_sidecar(destination, source_uri, fetched_by=fetched_by)
        written.append(destination)
    return written


def fetch_tess_products(
    workspace: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    exptime: int = 120,
) -> List[Product]:
    """Download SPOC light curves from MAST (network access required).

    Returns ``(local_path, source_uri)`` pairs staged in a temporary
    directory. The caller passes them to ``ingest_products``.
    """
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
        sector_value = None
        try:
            sector_value = int(row["sequence_number"]) or None
        except (KeyError, TypeError, ValueError):
            sector_value = None
        if sectors is not None and sector_value is not None and sector_value not in set(sectors):
            continue

        light_curve = search[index].download()
        fits_path = staging / "s{0:04d}_lc.fits".format(sector_value or index)
        light_curve.to_fits(path=fits_path, overwrite=True)
        obs_id = str(row["obs_id"]) if "obs_id" in row.colnames else str(row["productFilename"])
        source_uri = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{0}".format(
            obs_id
        )
        products.append((fits_path, source_uri))
    return products


def fetch_tess_tpfs(
    workspace: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    exptime: int = 120,
) -> List[Product]:
    """Download SPOC target pixel files from MAST (network access required).

    TPF products are staged as ``s{sec:04d}_tp.fits`` so the acquisition gate's
    provenance sidecar convention (``<stem>.provenance.json``) applies to them
    exactly like light curves — the ``tp`` stem marker is also what
    ``inputs.load_tpf_cubes`` uses to distinguish TPFs from light curves.
    """
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
        sector_value = None
        try:
            sector_value = int(row["sequence_number"]) or None
        except (KeyError, TypeError, ValueError):
            sector_value = None
        if sectors is not None and sector_value is not None and sector_value not in set(sectors):
            continue

        tpf = search[index].download()
        fits_path = staging / "s{0:04d}_tp.fits".format(sector_value or index)
        tpf.to_fits(str(fits_path), overwrite=True)
        obs_id = str(row["obs_id"]) if "obs_id" in row.colnames else str(row["productFilename"])
        source_uri = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{0}".format(
            obs_id
        )
        products.append((fits_path, source_uri))
    return products
