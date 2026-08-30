"""EXONYM command-line entry point.

Commands:
  init     Provision a candidate workspace from global templates
  list     List registered candidates (--phase, --tag, and classification filters)
  status   Show one candidate identity record
  track    Render the QVG progress telemetry dashboard
  advance  Validate the current gate and promote the workflow phase
  tag      Attach metadata tags to a candidate record
  freeze   Build a reproducibility bundle under releases/<version>/
  verify-release Verify a frozen bundle and replay its offline load boundary
  search   Run a BLS transit search on candidate light curve data
  screen   Run fixed-ephemeris photometric consistency checks
  plot     Generate diagnostic vetting figures for a candidate
  fetch-priors Fetch catalog parameters from ExoFOP and save to transit config
  verify   Audit source or candidate integrity with scoped cache-aware checks
  debug    Run candidate-free source diagnostics and synthetic regressions
   set-state      Set the lifecycle state without hand-editing candidate.json
  review         Record an evidence-backed classification review
   classify       Propose or apply conservative classification to candidates
  storage        Produce a read-only candidate storage inventory
  reject         Permanently block vetting with a validated decisive rejection
  analysis-status  Record analysis-stage coverage

Workflow automation:
  survey   Manage survey denominator, harvest, auto-vet, run-loop, and sensitivity
  engine   Inspect, validate, and execute analytical and vetting engines
  wizard   Interactive guided setup across the pipeline stages
  checkpoint  Snapshot, list, restore, or delete candidate analysis state
  catalog  Capture candidate-local evidence from reviewed catalog providers

Acquisition and data preparation:
  ingest   Download SPOC products and record provenance
  detrend  Write an opt-in candidate-local detrended light-curve artifact
  ds9-regions  Export validated archival Gaia sources as DS9 region file

Scientific analysis commands:
  asteroseismology  Oscillation envelope, Delta-nu, and seismic M*/R*
  localization      Sub-pixel PRF transit source localization
  sed               SED stellar atmosphere posterior fit
  fit               MCMC transit fit with free limb darkening
  phasecurve        Phase curve and secondary eclipse search
  ttv               Transit timing variation (O-C) analysis
  activity          Stellar rotation periodogram analysis
  dilution          Aperture robustness and dilution sensitivity
  archive           Query Gaia DR3 and NASA ExoFOP for archival vetting
  rv                Ingest candidate-local RV data and fit a Keplerian model
  vet               Run TRICERATOPS Monte Carlo FPP simulation
  triage            Run automated pre-vetting diagnostic evidence aggregation

Optional specialized-model adapters:
  planetsynth    Run opt-in giant-planet cooling interpretation
  pyppluss       Test a declared anomalous-transit hypothesis
  catwoman       Test a declared terminator-asymmetry hypothesis (Catwoman)
  squishyplanet  Test a declared terminator-asymmetry hypothesis (SquishyPlanet)

Paper export:
  export-paper    Export candidate evidence into a manuscript macro bundle

This module is intentionally a thin dispatcher. Candidate-specific inputs and
outputs are delegated to candidate-owning modules; this entry point does not
turn command success into a scientific claim.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .freeze import freeze, verify_release
from .gatekeeper import GateError, advance
from .isolation import add_verify_arguments, format_report, run_verify_command
from .tagging import add_tags, filter_candidates
from .tracking import candidate_telemetry, format_dashboard
from .workspace import (
    LIFECYCLE_STATES,
    PUBLICATION_STATES,
    RETENTION_CLASSES,
    REVIEW_STATUSES,
    SCIENTIFIC_DISPOSITIONS,
    WORKFLOW_PHASES,
    create_candidate,
    discover_candidates,
    discover_candidates_with_outcomes,
    load_candidate,
    workspace_layout,
)


def _default_repository_root() -> Path:
    """Choose a source checkout root or the caller's installed-workspace root.

    Returns:
        The checkout containing ``pyproject.toml`` when available; otherwise
        the resolved current working directory.
    """
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()


def _build_parser() -> argparse.ArgumentParser:
    """Build the complete ``exonym`` command-line argument parser.

    Returns:
        Parser with global options and all supported subcommands. Parsing and
        command execution remain the responsibility of :func:`main`.
    """
    parser = argparse.ArgumentParser(
        prog="exonym",
        description="EXONYM candidate framework: provision, gate, track, and freeze "
        "exoplanet candidate research workspaces.",
    )
    parser.add_argument("--version", action="version", version="exonym " + __version__)
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_repository_root(),
        help="Repository root containing the candidate directory.",
    )
    parser.add_argument(
        "--banner",
        action="store_true",
        default=False,
        help="Show the startup animation banner and exit.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        default=False,
        dest="no_animation",
        help="Skip the startup animation (also honoured by -q / --quiet).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress optional output including the startup animation.",
    )
    # required=False so that invoking 'exonym' with no subcommand reaches main()
    # and can show the banner + help instead of an argparse error.
    commands = parser.add_subparsers(dest="command", metavar="<command>", required=False)

    init_parser = commands.add_parser("init", help="Provision a candidate workspace.")
    init_parser.add_argument("candidate_id", help="Lowercase workspace identifier.")
    init_parser.add_argument("--toi", help="Canonical TOI identifier, without the TOI prefix.")
    init_parser.add_argument("--tic", help="Canonical TIC identifier.")
    init_parser.add_argument(
        "--mission",
        choices=["tess", "kepler", "k2", "plato", "cheops"],
        help="Originating mission for the target.",
    )
    init_parser.add_argument(
        "--tag", action="append", default=[], help="Attach a metadata tag (repeatable)."
    )

    list_parser = commands.add_parser("list", help="List registered candidates.")
    list_parser.add_argument(
        "--phase", choices=WORKFLOW_PHASES, help="Filter by workflow phase."
    )
    list_parser.add_argument("--tag", help="Filter by metadata tag.")
    list_parser.add_argument(
        "--mission",
        choices=["tess", "kepler", "k2", "plato", "cheops"],
        help="Filter by originating mission.",
    )
    list_parser.add_argument(
        "--disposition",
        choices=SCIENTIFIC_DISPOSITIONS,
        help="Filter by scientific disposition.",
    )
    list_parser.add_argument(
        "--publication",
        choices=PUBLICATION_STATES,
        help="Filter by publication state.",
    )
    list_parser.add_argument(
        "--lifecycle",
        choices=LIFECYCLE_STATES,
        help="Filter by lifecycle state.",
    )
    list_parser.add_argument(
        "--review-status",
        choices=REVIEW_STATUSES,
        help="Filter by human-review status.",
    )
    list_parser.add_argument(
        "--retention-class",
        choices=RETENTION_CLASSES,
        help="Filter by operational retention class.",
    )

    storage_parser = commands.add_parser(
        "storage", help="Produce a read-only filesystem inventory."
    )
    storage_commands = storage_parser.add_subparsers(dest="storage_action", required=True)
    storage_report_parser = storage_commands.add_parser(
        "report", help="Measure regular-file counts and bytes without reading contents."
    )
    storage_report_parser.add_argument(
        "candidate_id", nargs="?", help="Limit the report to one candidate workspace."
    )

    classify_parser = commands.add_parser(
        "classify", help="Propose or apply conservative administrative classification."
    )
    classify_parser.add_argument(
        "--candidate", dest="candidate_id", help="Limit classification to one candidate."
    )
    classify_mode = classify_parser.add_mutually_exclusive_group()
    classify_mode.add_argument(
        "--apply", action="store_true", help="Write the proposed classification reviews."
    )
    classify_mode.add_argument(
        "--verify", action="store_true", help="Verify classification review evidence hashes only."
    )

    organize_parser = commands.add_parser(
        "organize",
        help="Move candidates into lifecycle-group folders (dry-run by default).",
    )
    organize_parser.add_argument(
        "--candidate", dest="candidate_id", help="Limit organization to one candidate."
    )
    organize_parser.add_argument(
        "--by", choices=("lifecycle",), default="lifecycle",
        help="Grouping key (default: lifecycle).",
    )
    organize_mode = organize_parser.add_mutually_exclusive_group()
    organize_mode.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Preview proposed moves without changing the filesystem (default).",
    )
    organize_mode.add_argument(
        "--apply", action="store_true", help="Perform the filesystem moves.",
    )

    survey_parser = commands.add_parser(
        "survey", help="Create and operate a bounded independent-discovery survey."
    )
    survey_commands = survey_parser.add_subparsers(dest="survey_action", required=True)
    survey_init_parser = survey_commands.add_parser("init", help="Create a survey manifest.")
    survey_init_parser.add_argument("survey_id", help="Lowercase survey identifier.")
    survey_init_parser.add_argument("--mission", choices=("tess",), required=True)
    survey_init_parser.add_argument("--sectors", nargs="+", type=int, required=True)
    survey_init_parser.add_argument(
        "--review-snr", type=float, default=6.0,
        help="Preregistered BLS SNR for internal human-review routing.",
    )
    survey_add_parser = survey_commands.add_parser(
        "add-target", help="Register a TOI-free candidate workspace in the survey denominator."
    )
    survey_add_parser.add_argument("survey_id")
    survey_add_parser.add_argument("candidate_id")
    survey_search_parser = survey_commands.add_parser(
        "search", help="Run a sector-scoped BLS search for an eligible registered target."
    )
    survey_search_parser.add_argument("survey_id")
    survey_search_parser.add_argument("candidate_id")
    survey_sensitivity_parser = survey_commands.add_parser(
        "sensitivity",
        help="Run a fixed two-branch injection-recovery grid without changing survey routing.",
    )
    survey_sensitivity_parser.add_argument("survey_id")
    survey_sensitivity_parser.add_argument("candidate_id")
    survey_exclude_parser = survey_commands.add_parser(
        "exclude", help="Record a pre-search exclusion without changing candidate lifecycle."
    )
    survey_exclude_parser.add_argument("survey_id")
    survey_exclude_parser.add_argument("candidate_id")
    survey_exclude_parser.add_argument("--reason", required=True)
    survey_report_parser = survey_commands.add_parser(
        "report", help="Show the survey denominator and recorded outcomes."
    )
    survey_report_parser.add_argument("survey_id")
    survey_harvest_parser = survey_commands.add_parser(
        "harvest", help="Stream a supplied TCE CSV, retain live novelty evidence, and register eligible targets."
    )
    survey_harvest_parser.add_argument("survey_id")
    survey_harvest_parser.add_argument("--source", required=True, help="HTTPS or local CSV TCE release source.")
    survey_harvest_parser.add_argument("--max-candidates", type=int, default=25)
    survey_harvest_parser.add_argument("--minimum-snr", type=float, default=20.0)
    survey_harvest_parser.add_argument("--period-min", type=float, default=1.0)
    survey_harvest_parser.add_argument("--period-max", type=float, default=15.0)
    survey_harvest_parser.add_argument("--depth-min", type=float, default=200.0)
    survey_harvest_parser.add_argument("--depth-max", type=float, default=1500.0)
    survey_harvest_parser.add_argument("--radius-min", type=float, default=1.2)
    survey_harvest_parser.add_argument("--radius-max", type=float, default=3.5)
    survey_harvest_parser.add_argument("--stellar-radius-max", type=float, default=1.3)
    survey_harvest_parser.add_argument("--tmag-max", type=float, default=12.5)
    survey_harvest_parser.add_argument("--timeout", type=float, default=20.0)
    survey_harvest_parser.add_argument("--freshness-hours", type=float, default=24.0)
    survey_auto_vet_parser = survey_commands.add_parser(
        "auto-vet", help="Run bounded candidate-local evidence collection without changing workflow state or claims."
    )
    survey_auto_vet_parser.add_argument("candidate_id", nargs="?")
    survey_auto_vet_parser.add_argument("--all", action="store_true", help="Process every registered candidate workspace.")
    survey_auto_vet_parser.add_argument("--sectors", nargs="+", type=int, default=None)
    survey_auto_vet_parser.add_argument("--n-draws", type=int, default=2000)
    survey_auto_vet_parser.add_argument(
        "--fit-samples",
        type=int,
        default=2500,
        help="Production steps forwarded to the transit fit; convergence diagnostics govern adequacy.",
    )
    survey_auto_vet_parser.add_argument("--no-download", action="store_true")
    survey_loop_parser = survey_commands.add_parser(
        "run-loop", help="Run bounded harvest and candidate-local vetting cycles; never reports a validation claim."
    )
    survey_loop_parser.add_argument("survey_id")
    survey_loop_parser.add_argument("--source", required=True)
    survey_loop_parser.add_argument("--max-cycles", type=int, default=1)
    survey_loop_parser.add_argument("--max-candidates", type=int, default=25)
    survey_loop_parser.add_argument("--minimum-snr", type=float, default=20.0)
    survey_loop_parser.add_argument("--period-min", type=float, default=1.0)
    survey_loop_parser.add_argument("--period-max", type=float, default=15.0)
    survey_loop_parser.add_argument("--depth-min", type=float, default=200.0)
    survey_loop_parser.add_argument("--depth-max", type=float, default=1500.0)
    survey_loop_parser.add_argument("--radius-min", type=float, default=1.2)
    survey_loop_parser.add_argument("--radius-max", type=float, default=3.5)
    survey_loop_parser.add_argument("--stellar-radius-max", type=float, default=1.3)
    survey_loop_parser.add_argument("--tmag-max", type=float, default=12.5)
    survey_loop_parser.add_argument("--timeout", type=float, default=20.0)
    survey_loop_parser.add_argument("--freshness-hours", type=float, default=24.0)
    survey_loop_parser.add_argument("--n-draws", type=int, default=2000)
    survey_loop_parser.add_argument(
        "--fit-samples",
        type=int,
        default=2500,
        help="Production steps forwarded to the transit fit; convergence diagnostics govern adequacy.",
    )

    engine_parser = commands.add_parser(
        "engine", help="Inspect and validate analytical and vetting engines."
    )
    engine_commands = engine_parser.add_subparsers(dest="engine_action", required=True)
    engine_list_parser = engine_commands.add_parser(
        "list", help="List registered analytical engines and runtime availability."
    )
    engine_list_parser.add_argument(
        "--json", action="store_true", help="Format output as JSON."
    )
    engine_check_parser = engine_commands.add_parser(
        "check", help="Check installation and readiness of a named engine."
    )
    engine_check_parser.add_argument("engine_name", help="Canonical engine identifier.")
    engine_run_parser = engine_commands.add_parser(
        "run", help="Execute an analytical engine and record a candidate-local run manifest."
    )
    engine_run_parser.add_argument("engine_name", help="Canonical engine identifier.")
    engine_run_parser.add_argument("candidate_id", help="Target candidate identifier.")
    engine_run_parser.add_argument("--signal", default=None, help="Optional signal prior identifier.")
    engine_report_parser = engine_commands.add_parser(
        "report", help="Report candidate-local engine execution history."
    )
    engine_report_parser.add_argument("candidate_id", help="Target candidate identifier.")

    catalog_parser = commands.add_parser(
        "catalog", help="Capture candidate-local evidence from reviewed catalog providers."
    )
    catalog_commands = catalog_parser.add_subparsers(dest="catalog_action", required=True)
    catalog_fetch_parser = catalog_commands.add_parser(
        "fetch", help="Fetch allowlisted catalog templates into append-only retrieval records."
    )
    catalog_fetch_parser.add_argument("candidate_id", help="Target candidate identifier.")
    catalog_fetch_parser.add_argument(
        "--providers", nargs="+", required=True,
        help="Allowlisted providers: mast, gaia, simbad, vizier, nasa-exoplanet-archive, irsa, ztf, exofop, lamost-dr11, smoka, mast-hubble-jwst.",
    )
    catalog_refresh_parser = catalog_commands.add_parser(
        "refresh", help="Create new retrievals only for expired catalog evidence."
    )
    catalog_refresh_parser.add_argument("candidate_id", help="Target candidate identifier.")
    catalog_report_parser = catalog_commands.add_parser(
        "report", help="Report catalog availability, ambiguity, staleness, and citations."
    )
    catalog_report_parser.add_argument("candidate_id", help="Target candidate identifier.")
    catalog_match_parser = catalog_commands.add_parser(
        "match-ephemeris",
        help="Compare a candidate ephemeris with fresh supported known-signal catalog rows.",
    )
    catalog_match_parser.add_argument("candidate_id", help="Target candidate identifier.")
    catalog_match_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal configuration suffix (for example .01).",
    )
    catalog_record_ephemeris_parser = catalog_commands.add_parser(
        "record-ephemeris",
        help="Record one reviewed, raw-hash-bound BJD_TDB known-signal ephemeris.",
    )
    catalog_record_ephemeris_parser.add_argument("candidate_id", help="Target candidate identifier.")
    catalog_record_ephemeris_parser.add_argument("--record-id", required=True)
    catalog_record_ephemeris_parser.add_argument("--source-kind", required=True)
    catalog_record_ephemeris_parser.add_argument("--source-name", required=True)
    catalog_record_ephemeris_parser.add_argument("--source-uri", required=True)
    catalog_record_ephemeris_parser.add_argument("--raw-artifact", required=True)
    catalog_record_ephemeris_parser.add_argument("--period-days", required=True, type=float)
    catalog_record_ephemeris_parser.add_argument("--epoch-bjd-tdb", required=True, type=float)
    catalog_record_ephemeris_parser.add_argument("--duration-hours", required=True, type=float)
    catalog_record_ephemeris_parser.add_argument("--retrieved-at", required=True)
    catalog_record_ephemeris_parser.add_argument("--expires-at", required=True)

    triage_parser = commands.add_parser(
        "triage", help="Aggregate pre-vetting findings into an automated decision record."
    )
    triage_parser.add_argument("candidate_id", help="Target candidate identifier.")
    triage_parser.add_argument("--policy-id", default="default-pre-vetting-triage", help="Triage policy identifier.")
    triage_parser.add_argument("--policy-version", default="1.0.0", help="Triage policy version.")
    triage_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal configuration suffix (for example .01).",
    )

    analysis_status_parser = commands.add_parser(
        "analysis-status", help="Record candidate-local analysis coverage and unavailable stages."
    )
    analysis_status_parser.add_argument("candidate_id")

    rejection_parser = commands.add_parser(
        "record-rejection",
        help="Record candidate-local decisive evidence that makes TRICERATOPS inapplicable.",
    )
    rejection_parser.add_argument("candidate_id", help="Target candidate identifier.")
    rejection_parser.add_argument("--reason", required=True, help="Evidence-based reason to stop before vetting.")
    rejection_parser.add_argument(
        "--evidence",
        required=True,
        help="Existing candidate-local evidence path, relative to the candidate workspace.",
    )

    status_parser = commands.add_parser("status", help="Show one candidate record.")
    status_parser.add_argument("candidate_id")

    track_parser = commands.add_parser("track", help="Render the telemetry dashboard.")
    track_parser.add_argument("candidate_id")

    checkpoint_parser = commands.add_parser(
        "checkpoint",
        help="Snapshot, inspect, or restore candidate analysis state.",
    )
    checkpoint_commands = checkpoint_parser.add_subparsers(dest="checkpoint_action", required=True)
    checkpoint_save_parser = checkpoint_commands.add_parser(
        "save", help="Create a hash-bound snapshot of config/decisions/outputs state."
    )
    checkpoint_save_parser.add_argument("candidate_id")
    checkpoint_save_parser.add_argument(
        "--name", required=True, help="Short lowercase label for the restore point."
    )
    checkpoint_list_parser = checkpoint_commands.add_parser(
        "list", help="List available restore points with digests."
    )
    checkpoint_list_parser.add_argument("candidate_id")
    checkpoint_restore_parser = checkpoint_commands.add_parser(
        "restore", help="Atomically roll mutable workspace state back to a snapshot."
    )
    checkpoint_restore_parser.add_argument("candidate_id")
    checkpoint_restore_parser.add_argument("--id", required=True, help="Checkpoint id from 'checkpoint list'.")
    checkpoint_restore_parser.add_argument(
        "--yes", action="store_true", help="Assume yes; required when stdin is not interactive."
    )
    checkpoint_delete_parser = checkpoint_commands.add_parser(
        "delete", help="Remove one snapshot archive and its manifest."
    )
    checkpoint_delete_parser.add_argument("candidate_id")
    checkpoint_delete_parser.add_argument("--id", required=True, help="Checkpoint id from 'checkpoint list'.")

    wizard_parser = commands.add_parser(
        "wizard", help="Interactive guided setup across the pipeline stages."
    )
    wizard_parser.add_argument(
        "candidate_id", nargs="?", default=None, help="Existing candidate; omit to provision first."
    )

    advance_parser = commands.add_parser("advance", help="Promote the workflow phase.")
    advance_parser.add_argument("candidate_id")

    setstate_parser = commands.add_parser(
        "set-state", help="Set the lifecycle state (safe alternative to hand-editing candidate.json)."
    )
    setstate_parser.add_argument("candidate_id")
    setstate_parser.add_argument("--state", required=True, help="New lifecycle state.")
    setstate_parser.add_argument("--reason", default=None, help="Reason for the state change.")

    review_parser = commands.add_parser(
        "review", help="Record an evidence-backed classification review."
    )
    review_parser.add_argument("candidate_id")
    review_parser.add_argument("--reviewer", required=True, help="Human reviewer identifier.")
    review_parser.add_argument("--reason", required=True, help="Reason for the classification decision.")
    review_parser.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="Candidate-local evidence path; repeat for multiple files.",
    )
    review_parser.add_argument(
        "--disposition",
        choices=SCIENTIFIC_DISPOSITIONS,
        help="Set the scientific disposition.",
    )
    review_parser.add_argument(
        "--publication",
        choices=PUBLICATION_STATES,
        help="Set the publication state.",
    )
    review_parser.add_argument(
        "--review-status",
        choices=REVIEW_STATUSES,
        help="Set the human-review status.",
    )
    review_parser.add_argument(
        "--retention-class",
        choices=RETENTION_CLASSES,
        help="Set the operational retention class.",
    )

    tag_parser = commands.add_parser("tag", help="Attach tags to a candidate.")
    tag_parser.add_argument("candidate_id")
    tag_parser.add_argument("tags", nargs="+", help="Tags to attach.")

    freeze_parser = commands.add_parser("freeze", help="Build a reproducibility bundle.")
    freeze_parser.add_argument("candidate_id")
    freeze_parser.add_argument("--version", help="Release version directory name.")

    verify_release_parser = commands.add_parser(
        "verify-release", help="Verify bundle integrity and replay its offline source/workspace load."
    )
    verify_release_parser.add_argument("candidate_id")
    verify_release_parser.add_argument("--version", required=True, help="Release version directory name.")

    ingest_parser = commands.add_parser(
        "ingest", help="Download SPOC products and record provenance."
    )
    ingest_parser.add_argument("candidate_id")
    ingest_parser.add_argument(
        "--sectors", nargs="+", type=int, default=None, help="TESS sectors to fetch."
    )
    ingest_parser.add_argument("--exptime", type=int, default=120, help="Cadence in seconds.")
    ingest_parser.add_argument(
        "--products",
        choices=("lc", "tp", "both"),
        default="lc",
        help="SPOC product type: light curves (lc), target pixel files (tp), or both.",
    )
    ingest_parser.add_argument(
        "--provider",
        choices=("spoc",),
        default="spoc",
        help="MAST product provider. Only raw, provenance-bound SPOC products are supported.",
    )
    ingest_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Maximum concurrent download threads (default: 4).",
    )

    detrend_parser = commands.add_parser(
        "detrend", help="Write an opt-in candidate-local detrended light-curve artifact."
    )
    detrend_parser.add_argument("candidate_id")
    detrend_parser.add_argument(
        "--method", choices=("running-median", "wotan", "celerite"), default="running-median"
    )
    detrend_parser.add_argument(
        "--window-days", type=float, required=True, help="Positive detrending timescale in days."
    )
    detrend_parser.add_argument(
        "--no-transit-mask",
        action="store_true",
        help=(
            "Do not protect a known transit window. Use for blind discovery so a "
            "previous BLS ephemeris cannot suppress a different search peak."
        ),
    )

    ds9_parser = commands.add_parser(
        "ds9-regions", help="Export validated archival Gaia sources as a candidate-local DS9 region file."
    )
    ds9_parser.add_argument("candidate_id")

    verify_parser = commands.add_parser("verify", help="Audit source isolation or candidate integrity.")
    add_verify_arguments(verify_parser)

    debug_parser = commands.add_parser(
        "debug",
        help="Audit target-neutral code without reading or writing candidate workspaces.",
    )
    from .debugger import add_debug_arguments

    add_debug_arguments(debug_parser)

    export_paper_parser = commands.add_parser(
        "export-paper", help="Export candidate evidence into a candidate-local manuscript macro bundle."
    )
    export_paper_parser.add_argument("candidate_id", help="Target candidate identifier.")
    export_paper_parser.add_argument(
        "--signal", default=None, help="Optional per-signal fit and vetting artifact suffix."
    )

    search_parser = commands.add_parser("search", help="Run a transit search on candidate data.")
    search_parser.add_argument("candidate_id")
    search_parser.add_argument(
        "--period-min",
        type=float,
        default=None,
        help=(
            "Minimum blind-search orbital period (default 0.5 d). With --signal, omit it "
            "or match the prior-defined ±0.1 d window."
        ),
    )
    search_parser.add_argument(
        "--period-max",
        type=float,
        default=None,
        help=(
            "Maximum blind-search orbital period (default 15.0 d). With --signal, omit it "
            "or match the prior-defined ±0.1 d window."
        ),
    )
    search_parser.add_argument(
        "--engine",
        choices=("bls", "tls"),
        default="bls",
        help="Search engine: BLS or optional native-cadence Transit Least Squares.",
    )
    search_parser.add_argument(
        "--duration-grid-hours",
        nargs="+",
        type=float,
        default=None,
        metavar="HOURS",
        help=(
            "BLS-only blind-search transit-duration trials in hours. "
            "Cannot be combined with --signal."
        ),
    )
    search_parser.add_argument(
        "--signal",
        default=None,
        help="Targeted search using prior from config/signals/transit_config<signal>.json",
    )
    search_parser.add_argument(
        "--detrending-method",
        choices=("running-median", "wotan", "celerite"),
        default=None,
        help="Use a hash-bound candidate-local detrending product instead of direct FITS flux.",
    )

    screen_parser = commands.add_parser(
        "screen", help="Run fixed-ephemeris primary, odd-even, and half-phase screening."
    )
    screen_parser.add_argument("candidate_id")
    screen_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal transit config name (e.g. .01 -> config/signals/transit_config.01.json).",
    )
    screen_parser.add_argument(
        "--detrending-method",
        choices=("running-median", "wotan", "celerite"),
        default=None,
        help="Use a hash-bound candidate-local detrending product instead of direct FITS flux.",
    )

    plot_parser = commands.add_parser("plot", help="Generate diagnostic vetting plots.")
    plot_parser.add_argument("candidate_id")
    plot_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal transit config name used for phase folding (for example, .01).",
    )
    plot_parser.add_argument(
        "--corner",
        action="store_true",
        help="Generate an MCMC posterior corner plot from the matching fit chain.",
    )

    fetch_parser = commands.add_parser("fetch-priors", help="Fetch ExoFOP transit priors.")
    fetch_parser.add_argument("candidate_id")

    vet_parser = commands.add_parser(
        "vet", help="Run TRICERATOPS Monte Carlo FPP simulation on candidate."
    )
    vet_parser.add_argument("candidate_id")
    vet_parser.add_argument(
        "--n-draws", type=int, default=2000, help="Number of Monte Carlo draws."
    )
    vet_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal transit config name (e.g. .01 -> config/signals/transit_config.01.json).",
    )
    vet_parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Maximum Numba threads for vectorized TRICERATOPS likelihoods (default: 4).",
    )
    vet_parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Suppress the interactive telemetry HUD during long-running vetting.",
    )

    asteroseismology_parser = commands.add_parser(
        "asteroseismology", help="Estimate stellar oscillation envelope and seismic M*/R*."
    )
    asteroseismology_parser.add_argument("candidate_id")
    asteroseismology_parser.add_argument(
        "--numax-min", type=float, default=100.0, help="Minimum nu_max search bound in microHz."
    )
    asteroseismology_parser.add_argument(
        "--numax-max", type=float, default=1600.0, help="Maximum nu_max search bound in microHz."
    )

    localization_parser = commands.add_parser(
        "localization", help="Sub-pixel PRF transit source localization on TPFs."
    )
    localization_parser.add_argument("candidate_id")
    localization_parser.add_argument(
        "--search-radius", type=float, default=60.0,
        help="Gaia neighbor search radius in arcseconds.",
    )

    sed_parser = commands.add_parser(
        "sed", help="Fit stellar atmosphere posterior to broadband photometry."
    )
    sed_parser.add_argument("candidate_id")

    fit_parser = commands.add_parser(
        "fit", help="MCMC transit fit with free limb darkening and density locking."
    )
    fit_parser.add_argument("candidate_id")
    fit_parser.add_argument(
        "--n-samples",
        type=int,
        default=2500,
        help="Emcee production steps per walker; dynesty uses this to scale initial live points. GPU NUTS uses its own robust default. Convergence diagnostics govern adequacy.",
    )
    fit_parser.add_argument(
        "--detrending-method",
        choices=("running-median", "wotan", "celerite"),
        default=None,
        help="Use a hash-bound candidate-local detrending product instead of direct FITS flux.",
    )
    fit_parser.add_argument(
        "--sampler",
        choices=("auto", "emcee", "numpyro", "dynesty"),
        default="auto",
        help="Posterior sampler; auto selects CUDA NumPyro when available, otherwise emcee.",
    )
    fit_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Compute device for auto/NumPyro fitting; unavailable GPUs fall back to CPU emcee.",
    )
    fit_parser.add_argument(
        "--eccentric", action="store_true", help="Sample eccentric orbit parameters."
    )
    fit_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal transit config name (e.g. .01 -> config/signals/transit_config.01.json).",
    )
    fit_parser.add_argument(
        "--ldtk-prior",
        action="store_true",
        help="Use exactly one recorded candidate-local LDTk quadratic limb-darkening prior.",
    )
    fit_parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for ensemble likelihood evaluation (default 1).",
    )
    fit_parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Suppress the interactive telemetry HUD during long-running fitting.",
    )
    fit_parser.add_argument(
        "--resume",
        default=None,
        help="Path to a current Exonym no-pickle checkpoint .npz to resume from.",
    )

    phasecurve_parser = commands.add_parser(
        "phasecurve", help="Phase curve and secondary eclipse harmonic search."
    )
    phasecurve_parser.add_argument("candidate_id")

    ttv_parser = commands.add_parser(
        "ttv", help="Transit timing variation (O-C) analysis."
    )
    ttv_parser.add_argument("candidate_id")
    ttv_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal transit config name (e.g. .01 -> config/signals/transit_config.01.json).",
    )
    ttv_parser.add_argument(
        "--fit-orbital-decay",
        action="store_true",
        help="Include a formal quadratic-ephemeris period-derivative diagnostic.",
    )

    activity_parser = commands.add_parser(
        "activity", help="Stellar rotation GLS periodogram analysis."
    )
    activity_parser.add_argument("candidate_id")

    dilution_parser = commands.add_parser(
        "dilution", help="Aperture robustness and dilution sensitivity."
    )
    dilution_parser.add_argument("candidate_id")

    archive_parser = commands.add_parser(
        "archive", help="Query Gaia DR3 and NASA ExoFOP for candidate archival vetting."
    )
    archive_parser.add_argument("candidate_id")
    archive_parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=60.0,
        help="Gaia neighbor search radius in arcseconds (default: 60, sufficient for TESS crowding context).",
    )

    rv_parser = commands.add_parser(
        "rv", help="Ingest candidate-local radial velocities and fit descriptive Keplerian evidence."
    )
    rv_commands = rv_parser.add_subparsers(dest="rv_action", required=True)
    rv_ingest_parser = rv_commands.add_parser(
        "ingest", help="Validate and copy a finite RV observation JSON file into the candidate workspace."
    )
    rv_ingest_parser.add_argument("candidate_id")
    rv_ingest_parser.add_argument("source", type=Path, help="Observation JSON source file.")
    rv_fit_parser = rv_commands.add_parser(
        "fit", help="Compare constant and eccentric Keplerian RV models at a fixed period."
    )
    rv_fit_parser.add_argument("candidate_id")
    rv_fit_parser.add_argument("--period-days", type=float, required=True, help="Fixed orbital period in days.")
    rv_fit_parser.add_argument(
        "--period-uncertainty-days",
        type=float,
        default=None,
        help="Optional uncertainty of the fixed input period in days.",
    )

    planetsynth_parser = commands.add_parser(
        "planetsynth",
        help="Run opt-in giant-planet cooling interpretation from data/external/planetsynth_characterization.json.",
    )
    planetsynth_parser.add_argument("candidate_id")
    pyppluss_parser = commands.add_parser(
        "pyppluss",
        help="Test a declared anomalous-transit hypothesis from data/external/anomalous_transit_hypothesis.json.",
    )
    pyppluss_parser.add_argument("candidate_id")
    catwoman_parser = commands.add_parser(
        "catwoman",
        help="Test a declared terminator-asymmetry hypothesis with the optional Catwoman adapter.",
    )
    catwoman_parser.add_argument("candidate_id")
    squishyplanet_parser = commands.add_parser(
        "squishyplanet",
        help="Test a declared terminator-asymmetry hypothesis with the optional SquishyPlanet adapter.",
    )
    squishyplanet_parser.add_argument("candidate_id")
    return parser


def _print_json(value: object) -> None:
    """Print one value as deterministic, human-readable JSON.

    Args:
        value: JSON-serializable value to render to standard output.
    """
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def _harvest_filters(args: argparse.Namespace):
    """Build a survey-harvest filter contract from parsed CLI arguments.

    Args:
        args: Parsed command-line namespace for a harvest-capable survey
            subcommand.

    Returns:
        ``TceFilters`` populated from the explicit source-release bounds.
    """
    from .survey_harvest import TceFilters

    return TceFilters(
        minimum_snr=args.minimum_snr,
        period_min_days=args.period_min,
        period_max_days=args.period_max,
        depth_min_ppm=args.depth_min,
        depth_max_ppm=args.depth_max,
        radius_min_earth=args.radius_min,
        radius_max_earth=args.radius_max,
        stellar_radius_max_solar=args.stellar_radius_max,
        tmag_max=args.tmag_max,
    )


def _is_autonomous_batch_command(args: argparse.Namespace) -> bool:
    """Return whether a command needs a root-level autonomous incident record."""
    return args.command == "survey" and args.survey_action in {"harvest", "auto-vet", "run-loop"}


def _command_text(argv: Optional[Sequence[str]]) -> str:
    """Render the invoked command for an incident record."""
    import sys

    return shlex.join(["exonym", *(argv if argv is not None else sys.argv[1:])])


def _run_loop_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and dispatch one EXONYM operation.

    Args:
        argv: Optional argument sequence excluding the executable name. When
            omitted, arguments are read from the process command line.

    Returns:
        ``0`` for successful commands, or a command-specific nonzero status
        for completed checks that report failure conditions.

    Raises:
        SystemExit: For parser errors and operational exceptions rendered by
            ``argparse`` with exit status ``2``.

    Note:
        Dispatching an analysis command records or reports its implemented
        operation. It does not by itself validate a candidate or establish a
        scientific claim.
    """
    import sys as _sys

    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    parser = _build_parser()
    args = parser.parse_args(argv)
    repository_root = args.root.resolve()

    # --banner or no subcommand: show animation then clean categorized overview.
    _skip_anim = getattr(args, "no_animation", False) or getattr(args, "quiet", False)
    if getattr(args, "banner", False) or args.command is None:
        from .banner import print_cli_overview, run_banner

        run_banner(skip=_skip_anim or not _sys.stdout.isatty())
        print_cli_overview()
        return 0

    try:
        if args.command == "debug":
            from .debugger import format_debug_report, run_debug

            report = run_debug(
                repository_root,
                mode=args.debug_mode,
                since=args.since,
            )
            print(format_debug_report(report, args.debug_format))
            return report.exit_code

        if args.command == "verify":
            remediated, report = run_verify_command(
                repository_root,
                source=args.source,
                candidates=args.candidates,
                legacy_scope=args.scope,
                schemas_only=args.schemas_only,
                fix=args.fix,
                fresh=args.fresh,
                candidate_id=args.candidate_id,
            )
            if remediated is not None:
                _print_json({"remediated": remediated})
            print(format_report(report))
            return 0 if report.ok else 1

        if args.command == "export-paper":
            from .paper_export import export_paper

            candidate = load_candidate(repository_root, args.candidate_id)
            output = export_paper(candidate, signal=args.signal)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "init":
            candidate = create_candidate(
                repository_root,
                args.candidate_id,
                toi=args.toi,
                tic=args.tic,
                tags=args.tag or None,
                mission=args.mission,
            )
            _print_json(candidate.metadata)
            return 0

        if args.command == "list":
            candidates = filter_candidates(
                discover_candidates(repository_root),
                tag=args.tag,
                phase=args.phase,
                mission=args.mission,
                disposition=args.disposition,
                publication=args.publication,
                lifecycle=args.lifecycle,
                review_status=args.review_status,
                retention_class=args.retention_class,
            )
            _print_json([candidate.metadata for candidate in candidates])
            return 0

        if args.command == "storage":
            from .storage import build_storage_report

            if args.storage_action == "report":
                _print_json(build_storage_report(repository_root, args.candidate_id))
                return 0
            raise ValueError("unsupported storage action")

        if args.command == "classify":
            from .classification import batch_classify, verify_classification_records

            if args.verify:
                result = verify_classification_records(
                    repository_root, candidate_id=args.candidate_id
                )
                _print_json(result)
                return 0 if result["status"] == "pass" else 1
            _print_json(
                batch_classify(
                    repository_root,
                    candidate_id=args.candidate_id,
                    apply=args.apply,
                )
            )
            return 0

        if args.command == "organize":
            from .workspace import organize_candidates

            apply = getattr(args, "apply", False)
            result = organize_candidates(
                repository_root,
                candidate_id=args.candidate_id,
                by=args.by,
                apply=apply,
            )
            _print_json(result)
            return 0 if result["summary"]["errors"] == 0 else 1

        if args.command == "survey":
            from .survey import (
                create_survey,
                exclude_survey_target,
                load_survey,
                load_survey_candidate,
                register_survey_target,
                run_survey_sensitivity,
                run_survey_search,
                survey_summary,
            )

            if args.survey_action == "init":
                survey = create_survey(
                    repository_root, args.survey_id, args.mission, args.sectors, args.review_snr
                )
                _print_json(survey.metadata)
                return 0
            if args.survey_action == "auto-vet":
                from .autonomous import auto_vet_candidate, record_autonomous_incident

                if bool(args.all) == (args.candidate_id is not None):
                    raise ValueError("provide exactly one candidate_id or --all")
                if args.all:
                    candidates, outcomes = discover_candidates_with_outcomes(repository_root)
                else:
                    candidates = [load_survey_candidate(repository_root, args.candidate_id)]
                    outcomes = []
                for candidate in candidates:
                    try:
                        manifest = auto_vet_candidate(
                            candidate,
                            sectors=args.sectors,
                            n_draws=args.n_draws,
                            fit_samples=args.fit_samples,
                            download=not args.no_download,
                            incident_command=_command_text(argv),
                        )
                    except Exception as exc:
                        outcomes.append(
                            {
                                "candidate_id": candidate.candidate_id,
                                "status": "failed",
                                "reason": "Auto-vet could not start: {0}: {1}".format(
                                    type(exc).__name__, exc
                                ),
                            }
                        )
                        record_autonomous_incident(repository_root, _command_text(argv), exc)
                    else:
                        outcomes.append(
                            {
                                "candidate_id": candidate.candidate_id,
                                "status": "completed",
                                "manifest": manifest.relative_to(repository_root).as_posix(),
                            }
                        )
                _print_json({"outcomes": outcomes, "claim_eligible": False})
                return 0
            survey = load_survey(repository_root, args.survey_id)
            if args.survey_action == "add-target":
                candidate = load_survey_candidate(repository_root, args.candidate_id)
                output = register_survey_target(survey, candidate)
                print(output.relative_to(repository_root).as_posix())
                return 0
            if args.survey_action == "search":
                candidate = load_survey_candidate(repository_root, args.candidate_id)
                output = run_survey_search(survey, candidate)
                print(output.relative_to(repository_root).as_posix())
                return 0
            if args.survey_action == "sensitivity":
                candidate = load_survey_candidate(repository_root, args.candidate_id)
                output = run_survey_sensitivity(survey, candidate)
                print(output.relative_to(repository_root).as_posix())
                return 0
            if args.survey_action == "exclude":
                output = exclude_survey_target(survey, args.candidate_id, args.reason)
                print(output.relative_to(repository_root).as_posix())
                return 0
            if args.survey_action == "report":
                _print_json(survey_summary(survey))
                return 0
            if args.survey_action == "harvest":
                from .survey_harvest import harvest_tces

                outcomes = harvest_tces(
                    survey,
                    args.source,
                    _harvest_filters(args),
                    args.max_candidates,
                    novelty_timeout=args.timeout,
                    freshness_hours=args.freshness_hours,
                )
                _print_json(outcomes)
                return 0
            if args.survey_action == "run-loop":
                from .autonomous import (
                    auto_vet_candidate,
                    auto_vet_started,
                    create_run_loop_journal,
                    record_autonomous_incident,
                    write_run_loop_journal,
                )
                from .survey_harvest import harvest_tces

                if args.max_cycles < 1:
                    raise ValueError("max_cycles must be at least one")
                journal_path, journal = create_run_loop_journal(
                    survey,
                    {
                        "max_cycles": args.max_cycles,
                        "max_candidates": args.max_candidates,
                        "source": args.source,
                        "n_draws": args.n_draws,
                        "fit_samples": args.fit_samples,
                    },
                )
                try:
                    for index in range(args.max_cycles):
                        cycle = {
                            "cycle": index + 1,
                            "status": "running",
                            "started_at": _run_loop_timestamp(),
                            "completed_at": None,
                            "harvest": [],
                            "auto_vet": [],
                        }
                        journal["cycles"].append(cycle)
                        write_run_loop_journal(journal_path, journal)
                        outcomes = harvest_tces(
                            survey,
                            args.source,
                            _harvest_filters(args),
                            args.max_candidates,
                            novelty_timeout=args.timeout,
                            freshness_hours=args.freshness_hours,
                        )
                        cycle["harvest"] = outcomes
                        write_run_loop_journal(journal_path, journal)
                        for outcome in outcomes:
                            candidate_id = outcome.get("candidate_id")
                            status = outcome.get("status")
                            if candidate_id is None or status not in {"registered", "already-provisioned"}:
                                continue
                            try:
                                candidate = load_survey_candidate(repository_root, candidate_id)
                            except (FileNotFoundError, ValueError) as exc:
                                cycle["auto_vet"].append(
                                    {
                                        "candidate_id": candidate_id,
                                        "status": "incomplete",
                                        "reason": "Candidate workspace could not be loaded: {0}".format(exc),
                                    }
                                )
                                write_run_loop_journal(journal_path, journal)
                                continue
                            if status == "already-provisioned" and auto_vet_started(candidate):
                                cycle["auto_vet"].append(
                                    {"candidate_id": candidate_id, "status": "already-started"}
                                )
                                write_run_loop_journal(journal_path, journal)
                                continue
                            try:
                                manifest = auto_vet_candidate(
                                    candidate,
                                    n_draws=args.n_draws,
                                    fit_samples=args.fit_samples,
                                    incident_command=_command_text(argv),
                                )
                            except Exception as exc:
                                cycle["auto_vet"].append(
                                    {
                                        "candidate_id": candidate_id,
                                        "status": "failed",
                                        "reason": "Auto-vet could not start: {0}: {1}".format(
                                            type(exc).__name__, exc
                                        ),
                                    }
                                )
                                write_run_loop_journal(journal_path, journal)
                                record_autonomous_incident(repository_root, _command_text(argv), exc)
                                continue
                            else:
                                cycle["auto_vet"].append(
                                    {
                                        "candidate_id": candidate_id,
                                        "status": "completed",
                                        "manifest": manifest.relative_to(repository_root).as_posix(),
                                    }
                                )
                            write_run_loop_journal(journal_path, journal)
                        cycle["status"] = "completed"
                        cycle["completed_at"] = _run_loop_timestamp()
                        write_run_loop_journal(journal_path, journal)
                except KeyboardInterrupt:
                    if journal["cycles"] and journal["cycles"][-1]["status"] == "running":
                        journal["cycles"][-1]["status"] = "interrupted"
                        journal["cycles"][-1]["completed_at"] = _run_loop_timestamp()
                    journal["status"] = "interrupted"
                    journal["completed_at"] = _run_loop_timestamp()
                    write_run_loop_journal(journal_path, journal)
                    raise
                except Exception as exc:
                    if journal["cycles"] and journal["cycles"][-1]["status"] == "running":
                        journal["cycles"][-1]["status"] = "failed"
                        journal["cycles"][-1]["completed_at"] = _run_loop_timestamp()
                    journal["status"] = "failed"
                    journal["completed_at"] = _run_loop_timestamp()
                    journal["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                    write_run_loop_journal(journal_path, journal)
                    raise
                journal["status"] = "completed"
                journal["completed_at"] = _run_loop_timestamp()
                write_run_loop_journal(journal_path, journal)
                _print_json(
                    {
                        "cycles": journal["cycles"],
                        "journal": journal_path.relative_to(repository_root).as_posix(),
                        "claim_eligible": False,
                    }
                )
                return 0

        if args.command == "engine":
            import dataclasses
            from .engines import check_engine, iter_engines

            if args.engine_action == "list":
                engine_statuses = iter_engines()
                if getattr(args, "json", False):
                    _print_json([dataclasses.asdict(e) for e in engine_statuses])
                else:
                    header = f"{'Engine':<14} {'Capability':<18} {'Installed':<11} {'Version':<12} {'Group':<14}"
                    print(header)
                    print("-" * len(header))
                    for e in engine_statuses:
                        inst = "yes" if e.installed else "no"
                        ver = e.version or "-"
                        print(
                            f"{e.name[:14]:<14} {e.capability[:18]:<18} {inst:<11} "
                            f"{ver[:12]:<12} {e.optional_group[:14]:<14}"
                        )
                return 0

            if args.engine_action == "check":
                ready, message = check_engine(args.engine_name)
                print(message)
                return 0 if ready else 1

            if args.engine_action == "run":
                from .engines import run_engine

                cand = load_candidate(repository_root, args.candidate_id)
                manifest = run_engine(cand, args.engine_name, signal=getattr(args, "signal", None))
                print(manifest.relative_to(repository_root).as_posix())
                data = json.loads(manifest.read_text(encoding="utf-8"))
                return 0 if data.get("status") == "succeeded" else 1

            if args.engine_action == "report":
                from .engines import report_candidate_engines

                cand = load_candidate(repository_root, args.candidate_id)
                runs = report_candidate_engines(cand)
                _print_json(runs)
                return 0

        if args.command == "catalog":
            from .catalog_federation import catalog_report, fetch_catalog, refresh_catalog

            candidate = load_candidate(repository_root, args.candidate_id)
            if args.catalog_action == "fetch":
                manifests = fetch_catalog(candidate, args.providers)
                _print_json([path.relative_to(repository_root).as_posix() for path in manifests])
                return 0
            if args.catalog_action == "refresh":
                manifests = refresh_catalog(candidate)
                _print_json([path.relative_to(repository_root).as_posix() for path in manifests])
                return 0
            if args.catalog_action == "report":
                _print_json(catalog_report(candidate))
                return 0
            if args.catalog_action == "match-ephemeris":
                from .ephemeris_matching import match_known_signal_ephemerides

                output = match_known_signal_ephemerides(candidate, signal=args.signal)
                print(output.relative_to(repository_root).as_posix())
                return 0
            if args.catalog_action == "record-ephemeris":
                from .ephemeris_matching import record_known_signal_ephemeris

                output = record_known_signal_ephemeris(
                    candidate,
                    args.record_id,
                    args.source_kind,
                    args.source_name,
                    args.source_uri,
                    args.raw_artifact,
                    args.period_days,
                    args.epoch_bjd_tdb,
                    args.duration_hours,
                    args.retrieved_at,
                    args.expires_at,
                )
                print(output.relative_to(repository_root).as_posix())
                return 0

        if args.command == "rv":
            from .radial_velocity import fit_radial_velocity, ingest_radial_velocity_observations

            candidate = load_candidate(repository_root, args.candidate_id)
            if args.rv_action == "ingest":
                output = ingest_radial_velocity_observations(candidate, args.source)
            else:
                output = fit_radial_velocity(
                    candidate,
                    period_days=args.period_days,
                    period_uncertainty_days=args.period_uncertainty_days,
                )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command in ("planetsynth", "pyppluss", "catwoman", "squishyplanet"):
            from .specialized_models import (
                run_catwoman,
                run_planetsynth,
                run_pyppluss,
                run_squishyplanet,
            )

            candidate = load_candidate(repository_root, args.candidate_id)
            runners = {
                "planetsynth": run_planetsynth,
                "pyppluss": run_pyppluss,
                "catwoman": run_catwoman,
                "squishyplanet": run_squishyplanet,
            }
            result = runners[args.command](candidate)
            output = result.report_path or result.manifest_path
            print(output.relative_to(repository_root).as_posix())
            return 0 if result.status == "succeeded" else 1

        candidate = load_candidate(repository_root, args.candidate_id)

        if args.command == "triage":
            from .engines import run_automated_triage

            triage_path = run_automated_triage(
                candidate,
                policy_id=getattr(args, "policy_id", "default-pre-vetting-triage"),
                policy_version=getattr(args, "policy_version", "1.0.0"),
                signal=getattr(args, "signal", None),
            )
            print(triage_path.relative_to(repository_root).as_posix())
            return 0

        if args.command == "analysis-status":
            from .analysis_status import build_analysis_status

            candidate = load_candidate(repository_root, args.candidate_id)
            print(build_analysis_status(candidate).relative_to(repository_root).as_posix())
            return 0

        if args.command == "record-rejection":
            from .statistical_vetting import record_decisive_rejection

            output = record_decisive_rejection(candidate, args.reason, args.evidence)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "status":
            result = dict(candidate.metadata)
            result["paths"] = {
                name: str(path.relative_to(repository_root)).replace("\\", "/")
                for name, path in workspace_layout(candidate).items()
            }
            _print_json(result)
            return 0

        if args.command == "track":
            print(
                format_dashboard(
                    candidate, candidate_telemetry(candidate)
                )
            )
            return 0

        if args.command == "advance":
            event = advance(candidate)
            _print_json(event)
            return 0

        if args.command == "set-state":
            from .gatekeeper import set_lifecycle_state

            _print_json(set_lifecycle_state(candidate, args.state, reason=args.reason))
            return 0

        if args.command == "review":
            from .review import apply_classification_review

            output = apply_classification_review(
                candidate,
                reviewer=args.reviewer,
                reason=args.reason,
                evidence_paths=args.evidence,
                scientific_disposition=args.disposition,
                publication=args.publication,
                review_status=args.review_status,
                retention_class=args.retention_class,
            )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "tag":
            _print_json(add_tags(candidate, args.tags))
            return 0

        if args.command == "freeze":
            release_dir = freeze(candidate, version=args.version)
            print(release_dir.relative_to(repository_root).as_posix())
            return 0

        if args.command == "verify-release":
            _print_json(verify_release(candidate, version=args.version))
            return 0

        if args.command == "ingest":
            from contextlib import ExitStack

            from .ingest import fetch_tess_products, fetch_tess_tpfs, ingest_products

            _workers = getattr(args, "workers", 4)
            _quiet = getattr(args, "quiet", False)
            with ExitStack() as staging_batches:
                all_products = []
                if args.products in ("lc", "both"):
                    all_products.extend(
                        staging_batches.enter_context(
                            fetch_tess_products(
                                candidate,
                                sectors=args.sectors,
                                exptime=args.exptime,
                                provider=args.provider,
                                quiet=_quiet,
                                workers=_workers,
                            )
                        )
                    )
                if args.products in ("tp", "both"):
                    all_products.extend(
                        staging_batches.enter_context(
                            fetch_tess_tpfs(
                                candidate,
                                sectors=args.sectors,
                                exptime=args.exptime,
                                provider=args.provider,
                                quiet=_quiet,
                                workers=_workers,
                            )
                        )
                    )
                if not all_products:
                    print("no products found for the requested sectors")
                    return 0
                written = ingest_products(candidate, all_products)
                _print_json(
                    [str(path.relative_to(candidate.path)).replace("\\", "/") for path in written]
                )
                return 0

        if args.command == "detrend":
            from .detrending import detrend_candidate, transit_mask_from_ephemeris
            from .inputs import BTJD_TIME_SYSTEM, load_light_curve_table, load_transit_ephemeris

            table = load_light_curve_table(
                candidate, max_points=None, require_raw_provenance=True
            )
            if table is None:
                raise ValueError("detrending requires readable candidate-local light-curve data")
            if table.get("time_system") != BTJD_TIME_SYSTEM:
                raise ValueError("detrending requires BTJD_TDB candidate photometry")
            ephemeris = None
            transit_mask = None
            if not args.no_transit_mask:
                ephemeris = load_transit_ephemeris(candidate)
                transit_mask = transit_mask_from_ephemeris(table["time"], ephemeris)
            artifact = detrend_candidate(
                candidate,
                table["time"],
                table["flux"],
                flux_err=table["flux_err"],
                method=args.method,
                window_days=args.window_days,
                sector=table["sector"],
                input_products=[
                    {
                        "path": Path(path).relative_to(candidate.path).as_posix(),
                        "sha256": digest,
                    }
                    for path, digest in zip(table["input_files"], table["input_sha256s"])
                ],
                transit_mask=transit_mask,
                transit_mask_ephemeris=ephemeris,
            )
            _print_json(
                {
                    "artifact": artifact.artifact_path.relative_to(repository_root).as_posix(),
                    "manifest": artifact.manifest_path.relative_to(repository_root).as_posix(),
                }
            )
            return 0

        if args.command == "ds9-regions":
            from .ds9 import export_ds9_regions

            output = export_ds9_regions(candidate)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "fetch-priors":
            from .priors import fetch_exofop_priors

            written = fetch_exofop_priors(candidate)
            _print_json([str(path.relative_to(candidate.path)).replace("\\", "/") for path in written])
            return 0

        if args.command == "search":
            from .search import run_bls_on_candidate, write_bls_transit_config

            output = run_bls_on_candidate(
                candidate,
                period_min=args.period_min,
                period_max=args.period_max,
                signal=args.signal,
                engine=args.engine,
                duration_grid_hours=args.duration_grid_hours,
                detrending_method=args.detrending_method,
            )
            if args.engine == "bls" and args.signal is None:
                write_bls_transit_config(candidate, output)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "screen":
            from .screening import run_fixed_ephemeris_screen

            output = run_fixed_ephemeris_screen(
                candidate, signal=args.signal, detrending_method=args.detrending_method
            )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "plot":
            from .plotting import generate_candidate_plots

            generated = generate_candidate_plots(
                candidate, signal=args.signal, include_corner=getattr(args, "corner", False)
            )
            _print_json([str(path.relative_to(repository_root)).replace("\\", "/") for path in generated])
            return 0

        if args.command == "vet":
            from .vetting.tricera_parse import run_triceratops_simulation
            from contextlib import nullcontext
            import sys

            telemetry_context = nullcontext(None)
            if args.progress and sys.stdout.isatty():
                from .telemetry import LiveTelemetry

                telemetry_context = LiveTelemetry(
                    candidate.candidate_id,
                    repository_root,
                    step_name="TRICERATOPS Monte Carlo Vetting",
                )
            with telemetry_context as telemetry:
                def progress_callback(step, done=None, total=None):
                    if telemetry is None:
                        return
                    if done is not None and total is not None:
                        telemetry.report_progress(done, total)
                    elif done is None and total is None:
                        # Status update or error surfaced from TRICERATOPS output.
                        telemetry.note_text(step)
                    else:
                        telemetry.set_step(step)

                output = run_triceratops_simulation(
                    candidate,
                    n_draws=args.n_draws,
                    signal=args.signal,
                    n_jobs=args.n_jobs,
                    progress_callback=progress_callback if telemetry is not None else None,
                )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "asteroseismology":
            from .asteroseismology import run_asteroseismology

            output = run_asteroseismology(
                candidate, numax_min_uhz=args.numax_min, numax_max_uhz=args.numax_max
            )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "localization":
            from .localization import run_prf_localization

            output = run_prf_localization(candidate, search_radius_arcsec=args.search_radius)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "sed":
            from .sed import run_sed_fit

            output = run_sed_fit(candidate)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "fit":
            import sys as _fit_sys
            from contextlib import nullcontext

            from .transit_fit import run_mcmc_transit_fit

            fit_kwargs = dict(
                n_samples=args.n_samples,
                eccentric=args.eccentric,
                signal=args.signal,
                use_ldtk_prior=args.ldtk_prior,
                sampler=args.sampler,
                device=args.device,
                detrending_method=args.detrending_method,
                n_jobs=args.n_jobs,
                resume=args.resume,
            )
            telemetry_context = nullcontext(None)
            if args.progress and _fit_sys.stdout.isatty():
                from .telemetry import LiveTelemetry

                telemetry_context = LiveTelemetry(
                    candidate.candidate_id,
                    repository_root=repository_root,
                    step_name="MCMC transit fit",
                )
            with telemetry_context as hud:
                if hud is not None:
                    def progress_callback(done, total=None, **metadata):
                        if total is None:
                            hud.report_evidence(done, **metadata)
                        else:
                            burn_in = int(metadata.pop("burn_in", 0))
                            production = int(metadata.pop("production", total))
                            hud.report_mcmc(
                                done,
                                total,
                                burn_in=burn_in,
                                production=production,
                                **metadata,
                            )
                    output = run_mcmc_transit_fit(
                        candidate,
                        progress=False,
                        progress_callback=progress_callback,
                        **fit_kwargs,
                    )
                else:
                    output = run_mcmc_transit_fit(candidate, progress=args.progress, **fit_kwargs)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "phasecurve":
            from .phasecurve import run_phase_curve_search

            output = run_phase_curve_search(candidate)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "ttv":
            from .ttv import run_ttv_analysis

            output = run_ttv_analysis(
                candidate,
                signal=args.signal,
                fit_orbital_decay=args.fit_orbital_decay,
            )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "activity":
            from .activity import run_stellar_activity

            output = run_stellar_activity(candidate)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "dilution":
            from .dilution import run_dilution_sensitivity

            output = run_dilution_sensitivity(candidate)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "archive":
            from .archive import run_archival_vetting

            output = run_archival_vetting(candidate, radius_arcsec=args.radius_arcsec)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "checkpoint":
            from . import checkpoints

            candidate_ws = load_candidate(repository_root, args.candidate_id)
            action = args.checkpoint_action
            if action == "save":
                manifest_path = checkpoints.save_checkpoint(candidate_ws, args.name)
                print(manifest_path.relative_to(repository_root).as_posix())
                return 0
            if action == "list":
                records = checkpoints.list_checkpoints(candidate_ws)
                if not records:
                    print("no checkpoints recorded for {0}".format(args.candidate_id))
                    return 0
                try:
                    from rich.table import Table

                    table = Table(title="Candidate checkpoints")
                    for column in (
                        "Checkpoint ID",
                        "Label",
                        "Lifecycle",
                        "Created (UTC)",
                        "Archive Size",
                        "SHA-256",
                    ):
                        table.add_column(column)
                    for record in records:
                        digest = record["archive"]["sha256"]
                        table.add_row(
                            record["checkpoint_id"],
                            record["label"],
                            record["lifecycle_state"],
                            record["created_utc"],
                            checkpoints.format_archive_size(record["archive"]["bytes"]),
                            digest[:12] + "...",
                        )
                    from rich.console import Console

                    Console().print(table)
                except ImportError:
                    for record in records:
                        print(
                            "{0}  {1}  {2}  {3}  {4}  {5}".format(
                                record["checkpoint_id"],
                                record["label"],
                                record["lifecycle_state"],
                                record["created_utc"],
                                record["archive"]["bytes"],
                                record["archive"]["sha256"],
                            )
                        )
                return 0
            if action == "restore":
                summary = checkpoints.restore_checkpoint(
                    candidate_ws, args.id, assume_yes=args.yes
                )
                _print_json(summary)
                return 0
            if action == "delete":
                checkpoints.delete_checkpoint(candidate_ws, args.id)
                print("deleted checkpoint {0}".format(args.id))
                return 0

        if args.command == "wizard":
            from .wizard import run_wizard

            return run_wizard(repository_root, args.candidate_id)
    except KeyboardInterrupt as exc:
        if _is_autonomous_batch_command(args):
            from .autonomous import record_autonomous_incident

            try:
                record_autonomous_incident(repository_root, _command_text(argv), exc)
            except OSError:
                pass
        raise
    except (FileExistsError, FileNotFoundError, ValueError, GateError, RuntimeError) as exc:
        if _is_autonomous_batch_command(args):
            from .autonomous import record_autonomous_incident

            try:
                record_autonomous_incident(repository_root, _command_text(argv), exc)
            except OSError:
                pass
        parser.exit(2, "error: {0}\n".format(exc))
    except Exception as exc:
        if not _is_autonomous_batch_command(args):
            raise
        from .autonomous import record_autonomous_incident

        try:
            record_autonomous_incident(repository_root, _command_text(argv), exc)
        except OSError:
            pass
        parser.exit(2, "error: unexpected failure: {0}\n".format(exc))

    parser.exit(2, "error: unknown command\n")


if __name__ == "__main__":
    raise SystemExit(main())
