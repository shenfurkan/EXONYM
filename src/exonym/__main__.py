"""EXONYM command-line entry point.

Commands:
  init     Provision a candidate workspace from global templates
  list     List registered candidates (--phase, --tag filters)
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
  verify   Run the repository isolation audit

Scientific analysis commands:
  asteroseismology  Oscillation envelope, Delta-nu, and seismic M*/R*
  localization      Sub-pixel PRF transit source localization
  sed               SED stellar atmosphere posterior fit
  fit               MCMC transit fit with free limb darkening
  phasecurve        Phase curve and secondary eclipse search
  ttv               Transit timing variation (O-C) analysis
  activity          Stellar rotation periodogram analysis
  dilution          Aperture robustness and dilution sensitivity
  archive           Query Gaia EDR3 and NASA ExoFOP for archival vetting
  rv                Ingest candidate-local RV data and fit a Keplerian model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .freeze import freeze, verify_release
from .gatekeeper import GateError, advance
from .isolation import format_report, run_audit
from .tagging import add_tags, filter_candidates
from .tracking import candidate_telemetry, format_dashboard
from .workspace import (
    create_candidate,
    discover_candidates,
    load_candidate,
    workspace_layout,
)


def _default_repository_root() -> Path:
    """Choose the source checkout root, or the caller's workspace when installed."""
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()


def _build_parser() -> argparse.ArgumentParser:
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
    commands = parser.add_subparsers(dest="command", required=True)

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
    list_parser.add_argument("--phase", help="Filter by workflow phase.")
    list_parser.add_argument("--tag", help="Filter by metadata tag.")
    list_parser.add_argument(
        "--mission",
        choices=["tess", "kepler", "k2", "plato", "cheops"],
        help="Filter by originating mission.",
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

    advance_parser = commands.add_parser("advance", help="Promote the workflow phase.")
    advance_parser.add_argument("candidate_id")

    setstate_parser = commands.add_parser(
        "set-state", help="Set the lifecycle state (safe alternative to hand-editing candidate.json)."
    )
    setstate_parser.add_argument("candidate_id")
    setstate_parser.add_argument("--state", required=True, help="New lifecycle state.")
    setstate_parser.add_argument("--reason", default=None, help="Reason for the state change.")

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
        choices=("spoc", "tesscut"),
        default="spoc",
        help="MAST product provider. TESSCut supports light curves only.",
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

    ds9_parser = commands.add_parser(
        "ds9-regions", help="Export validated archival Gaia sources as a candidate-local DS9 region file."
    )
    ds9_parser.add_argument("candidate_id")

    verify_parser = commands.add_parser("verify", help="Run the repository audit.")
    verify_parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Validate JSON schemas only (skip the isolation scan).",
    )

    search_parser = commands.add_parser("search", help="Run a transit search on candidate data.")
    search_parser.add_argument("candidate_id")
    search_parser.add_argument("--period-min", type=float, default=0.5, help="Minimum orbital period.")
    search_parser.add_argument("--period-max", type=float, default=15.0, help="Maximum orbital period.")
    search_parser.add_argument(
        "--engine",
        choices=("bls", "tls"),
        default="bls",
        help="Search engine: BLS or optional native-cadence Transit Least Squares.",
    )
    search_parser.add_argument(
        "--signal",
        default=None,
        help="Targeted search using prior from config/signals/transit_config<signal>.json",
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
        default=5000,
        help="Emcee production steps per walker or dynesty maximum likelihood calls.",
    )
    fit_parser.add_argument(
        "--sampler",
        choices=("emcee", "dynesty"),
        default="emcee",
        help="Posterior sampler; dynesty uses its optional inference dependency.",
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

    activity_parser = commands.add_parser(
        "activity", help="Stellar rotation GLS periodogram analysis."
    )
    activity_parser.add_argument("candidate_id")

    dilution_parser = commands.add_parser(
        "dilution", help="Aperture robustness and dilution sensitivity."
    )
    dilution_parser.add_argument("candidate_id")

    archive_parser = commands.add_parser(
        "archive", help="Query Gaia EDR3 and NASA ExoFOP for candidate archival vetting."
    )
    archive_parser.add_argument("candidate_id")
    archive_parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=10.0,
        help="Gaia neighbor search radius in arcseconds.",
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
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repository_root = args.root.resolve()

    try:
        if args.command == "verify":
            if args.schemas_only:
                from .isolation import IsolationReport

                from .schemas import validate_schemas

                report = IsolationReport()
                validate_schemas(repository_root, report)
            else:
                report = run_audit(repository_root)
            print(format_report(report))
            return 0 if report.ok else 1

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
            )
            _print_json([candidate.metadata for candidate in candidates])
            return 0

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
                        print(f"{e.name:<14} {e.capability:<18} {inst:<11} {ver:<12} {e.optional_group:<14}")
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

        if args.command in ("planetsynth", "pyppluss"):
            from .specialized_models import run_planetsynth, run_pyppluss

            candidate = load_candidate(repository_root, args.candidate_id)
            result = run_planetsynth(candidate) if args.command == "planetsynth" else run_pyppluss(candidate)
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
            from .ingest import fetch_tess_products, fetch_tess_tpfs, ingest_products

            all_products = []
            if args.products in ("lc", "both"):
                all_products.extend(
                    fetch_tess_products(
                        candidate, sectors=args.sectors, exptime=args.exptime, provider=args.provider
                    )
                )
            if args.products in ("tp", "both"):
                all_products.extend(
                    fetch_tess_tpfs(
                        candidate, sectors=args.sectors, exptime=args.exptime, provider=args.provider
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
            from .detrending import detrend_candidate
            from .inputs import load_light_curve_table

            table = load_light_curve_table(candidate, max_points=None)
            if table is None:
                raise ValueError("detrending requires readable candidate-local light-curve data")
            artifact = detrend_candidate(
                candidate,
                table["time"],
                table["flux"],
                flux_err=table["flux_err"],
                method=args.method,
                window_days=args.window_days,
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
            from .search import run_bls_on_candidate

            output = run_bls_on_candidate(
                candidate,
                period_min=args.period_min,
                period_max=args.period_max,
                signal=args.signal,
                engine=args.engine,
            )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "screen":
            from .screening import run_fixed_ephemeris_screen

            output = run_fixed_ephemeris_screen(candidate, signal=args.signal)
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

            output = run_triceratops_simulation(
                candidate, n_draws=args.n_draws, signal=args.signal
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
            from .transit_fit import run_mcmc_transit_fit

            output = run_mcmc_transit_fit(
                candidate,
                n_samples=args.n_samples,
                eccentric=args.eccentric,
                signal=args.signal,
                use_ldtk_prior=args.ldtk_prior,
                sampler=args.sampler,
            )
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "phasecurve":
            from .phasecurve import run_phase_curve_search

            output = run_phase_curve_search(candidate)
            print(output.relative_to(repository_root).as_posix())
            return 0

        if args.command == "ttv":
            from .ttv import run_ttv_analysis

            output = run_ttv_analysis(candidate, signal=args.signal)
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
    except (FileExistsError, FileNotFoundError, ValueError, GateError, RuntimeError) as exc:
        parser.exit(2, "error: {0}\n".format(exc))

    parser.exit(2, "error: unknown command\n")


if __name__ == "__main__":
    raise SystemExit(main())
