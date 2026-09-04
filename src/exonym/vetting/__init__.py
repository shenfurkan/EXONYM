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

Units and verified provenance
-----------------------------
Odd/even depth statistics use a shared declared depth unit (normally ppm) and
return dimensionless sigma; centroid offsets/errors are arcsec and return a
dimensionless score/probability; ellipsoidal inputs are named solar/AU/K/degree
quantities and return ppm.  Difference-image context is Bryson et al. (2013),
ADS ``2013PASP..125..889B``, DOI ``10.1086/671767``; leading ellipsoidal context
is Morris (1985), ADS ``1985ApJ...295..143M``, DOI ``10.1086/163359``.  Invalid
or incomplete inputs are unresolved, never passing, and no exported helper can
set ``claim_eligible``.
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
