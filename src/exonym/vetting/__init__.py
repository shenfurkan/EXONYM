"""Target-neutral diagnostic primitives used by the vetting workflow.

The functions exported here calculate individual screening statistics such as
odd/even depth consistency, a centroid-offset significance, and a leading-order
ellipsoidal-variation estimate. They are evidence inputs, not dispositions:
their candidate-local reports must still pass provenance, data-suitability, and
workflow gates before any downstream routing can use them.

Scientific boundary:
    These helpers do not calibrate a scene model or a population false-positive
    probability. A passing diagnostic therefore never establishes a discovery
    or validation claim on its own.
"""

from .centroid import centroid_gate, centroid_offset_pvalue, centroid_offset_z
from .ellipsoidal import ellipsoidal_gate, ellipsoidal_variation_amplitude_ppm
from .oddeven import odd_even_z
from .tricera_parse import fpp_gate, load_fpp_report
from .trex import TargetScene, TrexResult, run_trex_vetting

__all__ = [
    "centroid_gate",
    "centroid_offset_pvalue",
    "centroid_offset_z",
    "ellipsoidal_gate",
    "ellipsoidal_variation_amplitude_ppm",
    "odd_even_z",
    "fpp_gate",
    "load_fpp_report",
    "TargetScene",
    "TrexResult",
    "run_trex_vetting",
]
