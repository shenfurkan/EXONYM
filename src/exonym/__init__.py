"""EXONYM - Exoplanet Naming, Observation, and Yield Verification Management.

Target-neutral infrastructure for evidence-first exoplanet candidate
research. Every candidate lives under ``candidate/<candidate-id>/``; shared
code in this package never contains target constants.
"""

__version__ = "1.0.1"

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
