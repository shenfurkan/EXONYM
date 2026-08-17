"""Target-neutral engine registry and runtime capability catalog.

Provides capability descriptors, optional group mappings, and runtime
availability checks for analytical and vetting engines used by EXONYM.
Contains no candidate constants, sector numbers, or target identifiers.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EngineDescriptor:
    """Static registration descriptor for an analytical or vetting engine."""

    name: str
    capability: str
    optional_group: str
    module_name: str
    description: str


@dataclass(frozen=True)
class EngineStatus:
    """Resolved runtime status for an engine."""

    name: str
    capability: str
    optional_group: str
    module_name: str
    description: str
    installed: bool
    version: Optional[str]


# Target-neutral canonical catalog of Exonym engines
_ENGINE_CATALOG: Tuple[EngineDescriptor, ...] = (
    EngineDescriptor(
        name="bls",
        capability="search",
        optional_group="core",
        module_name="astropy.timeseries",
        description="Box Least Squares transit search via Astropy.",
    ),
    EngineDescriptor(
        name="tls",
        capability="search",
        optional_group="discovery",
        module_name="transitleastsquares",
        description="Transit Least Squares native-cadence search engine.",
    ),
    EngineDescriptor(
        name="batman",
        capability="fitting",
        optional_group="core",
        module_name="batman",
        description="Mandel-Agol transit light curve modeler.",
    ),
    EngineDescriptor(
        name="emcee",
        capability="sampler",
        optional_group="core",
        module_name="emcee",
        description="Affine-invariant ensemble MCMC sampler.",
    ),
    EngineDescriptor(
        name="dynesty",
        capability="sampler",
        optional_group="optional",
        module_name="dynesty",
        description="Dynamic nested sampling for Bayesian model comparison.",
    ),
    EngineDescriptor(
        name="triceratops",
        capability="vetting",
        optional_group="screening",
        module_name="triceratops",
        description="Bayesian transit validation and false positive probability calculation.",
    ),
    EngineDescriptor(
        name="pysyd",
        capability="asteroseismology",
        optional_group="asteroseismology",
        module_name="pysyd",
        description="Automated asteroseismic pipeline for solar-like oscillations.",
    ),
    EngineDescriptor(
        name="celerite",
        capability="detrending",
        optional_group="core",
        module_name="celerite",
        description="Fast 1D Gaussian Process regression for light curve modeling.",
    ),
    EngineDescriptor(
        name="wotan",
        capability="detrending",
        optional_group="optional",
        module_name="wotan",
        description="Comprehensive light curve detrending algorithms.",
    ),
    EngineDescriptor(
        name="ldtk",
        capability="priors",
        optional_group="core",
        module_name="ldtk",
        description="Limb Darkening Toolkit for stellar atmosphere profiles.",
    ),
    EngineDescriptor(
        name="corner",
        capability="plotting",
        optional_group="core",
        module_name="corner",
        description="Corner plot visualization for multidimensional posterior distributions.",
    ),
)


def _get_module_version(module_name: str) -> Optional[str]:
    """Retrieve installed version of a top-level package or module."""
    package_name = module_name.split(".")[0]
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        pass

    try:
        mod = importlib.import_module(package_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def get_engine_status(descriptor: EngineDescriptor) -> EngineStatus:
    """Inspect system runtime to determine availability of an engine."""
    package_name = descriptor.module_name.split(".")[0]
    spec = importlib.util.find_spec(package_name)
    installed = spec is not None
    version = _get_module_version(descriptor.module_name) if installed else None

    return EngineStatus(
        name=descriptor.name,
        capability=descriptor.capability,
        optional_group=descriptor.optional_group,
        module_name=descriptor.module_name,
        description=descriptor.description,
        installed=installed,
        version=version,
    )


def iter_engines() -> List[EngineStatus]:
    """List all registered engines with their current runtime installation status."""
    return [get_engine_status(desc) for desc in _ENGINE_CATALOG]


def get_engine(name: str) -> Optional[EngineStatus]:
    """Look up a specific engine by canonical name."""
    normalized = name.strip().lower()
    for desc in _ENGINE_CATALOG:
        if desc.name == normalized:
            return get_engine_status(desc)
    return None


def check_engine(name: str) -> Tuple[bool, str]:
    """Validate runtime readiness of a named engine.

    Returns:
        Tuple of (is_ready: bool, message: str).
    """
    status = get_engine(name)
    if status is None:
        valid_names = ", ".join(d.name for d in _ENGINE_CATALOG)
        return False, f"Unknown engine '{name}'. Supported engines: {valid_names}"

    if not status.installed:
        if status.optional_group == "core":
            dep_hint = "pip install -e ."
        elif status.optional_group == "optional":
            dep_hint = "pip install {0}".format(status.name)
        else:
            dep_hint = "pip install -e '.[{0}]'".format(status.optional_group)
        return (
            False,
            f"Engine '{status.name}' ({status.module_name}) is not installed. Install with: {dep_hint}",
        )

    ver_str = f" v{status.version}" if status.version else ""
    return True, f"Engine '{status.name}' ({status.capability}){ver_str} is installed and ready."
