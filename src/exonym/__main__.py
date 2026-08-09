"""EXONYM command-line entry point.

Commands:
  init     Provision a candidate workspace from global templates
  list     List registered candidates (--phase, --tag filters)
  status   Show one candidate identity record
  track    Render the QVG progress telemetry dashboard
  advance  Validate the current gate and promote the workflow phase
  tag      Attach metadata tags to a candidate record
  freeze   Build a reproducibility bundle under releases/<version>/
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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .freeze import freeze
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

    verify_parser = commands.add_parser("verify", help="Run the repository audit.")
    verify_parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Validate JSON schemas only (skip the isolation scan).",
    )

    search_parser = commands.add_parser("search", help="Run BLS transit search on candidate data.")
    search_parser.add_argument("candidate_id")
    search_parser.add_argument("--period-min", type=float, default=0.5, help="Minimum orbital period.")
    search_parser.add_argument("--period-max", type=float, default=15.0, help="Maximum orbital period.")
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
        "--n-samples", type=int, default=5000, help="MCMC production steps per walker."
    )
    fit_parser.add_argument(
        "--eccentric", action="store_true", help="Sample eccentric orbit parameters."
    )
    fit_parser.add_argument(
        "--signal",
        default=None,
        help="Per-signal transit config name (e.g. .01 -> config/signals/transit_config.01.json).",
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

        candidate = load_candidate(repository_root, args.candidate_id)

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

        if args.command == "ingest":
            from .ingest import fetch_tess_products, fetch_tess_tpfs, ingest_products

            all_products = []
            if args.products in ("lc", "both"):
                all_products.extend(
                    fetch_tess_products(candidate, sectors=args.sectors, exptime=args.exptime)
                )
            if args.products in ("tp", "both"):
                all_products.extend(
                    fetch_tess_tpfs(candidate, sectors=args.sectors, exptime=args.exptime)
                )
            if not all_products:
                print("no products found for the requested sectors")
                return 0
            written = ingest_products(candidate, all_products)
            _print_json(
                [str(path.relative_to(candidate.path)).replace("\\", "/") for path in written]
            )
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

            generated = generate_candidate_plots(candidate, signal=args.signal)
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
