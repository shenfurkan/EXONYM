"""EXONYM: evidence-first infrastructure for exoplanet-candidate research.

The package deliberately separates target-neutral implementation from
candidate-owned observations, decisions, and products. ``CandidateWorkspace``
provides the boundary used by every public command: shared code supplies
validation and workflow machinery, while candidate-specific measurements remain
below the corresponding workspace.

Scientific boundary:
    A structurally valid workflow record is not a scientific validation. The
    analysis gate remains closed until provenance-bound observed photometry and
    calibrated scene constraints can support an evidence-based claim path.

The module re-exports the small workspace API intended for library consumers.
It does not import optional scientific engines, so inspecting the package
version or workspace helpers remains possible in a minimal installation.
"""

__version__ = "1.5.0"

from .workspace import (
    CandidateWorkspace,
    create_candidate,
    discover_candidates,
    load_candidate,
    workspace_layout,
)

__all__ = [
    "__version__",
    "CandidateWorkspace",
    "create_candidate",
    "discover_candidates",
    "load_candidate",
    "workspace_layout",
]
