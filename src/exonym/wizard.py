"""Interactive guided pipeline wizard for novice operators.

The wizard collects parameters for the seven-phase workflow via
``rich.prompt`` primitives, validates every value before use, shows a
summary panel before executing anything, and then invokes the ordinary
``exonym`` subcommands through :func:`exonym.__main__.main` so wizard-driven
runs are byte-for-byte equivalent to hand-typed CLI invocations.

Isolation: this module holds no target identifiers, sector numbers, or
ephemeris values; everything is operator-supplied at runtime.

Scientific boundary: running wizard steps executes the same evidence
pipeline as the corresponding commands; it never relaxes gates, claims,
or validation semantics.
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Sequence, Tuple

DETREND_METHODS: Tuple[Tuple[str, str], ...] = (
    ("running-median", "running-median (fast standard)"),
    ("wotan", "wotan (spline/biweight, optional extra)"),
    ("celerite", "celerite (Gaussian process)"),
)
SEARCH_ENGINES: Tuple[str, ...] = ("bls", "tls")
PRODUCT_CHOICES: Tuple[str, ...] = ("lc", "tp", "both")
MISSIONS: Tuple[str, ...] = ("tess", "kepler", "k2", "plato", "cheops")

_DEFAULT_WINDOW_DAYS = 0.75
_DEFAULT_PERIOD_MIN = 0.5
_DEFAULT_PERIOD_MAX = 15.0
_DEFAULT_FIT_SAMPLES = 2500
_DEFAULT_VET_DRAWS = 2000


# ---------------------------------------------------------------------------
# Pure argv builders -- the tested seam between prompts and the real CLI.
# ---------------------------------------------------------------------------


def build_init_argv(
    candidate_id: str,
    mission: str,
    tic: Optional[str] = None,
    tags: Sequence[str] = (),
) -> List[str]:
    """Build ``init`` arguments from wizard answers."""
    argv = ["init", candidate_id, "--mission", mission]
    if tic:
        argv.extend(["--tic", tic])
    for tag in tags:
        argv.extend(["--tag", tag])
    return argv


def build_ingest_argv(candidate_id: str, sectors: Sequence[int], products: str) -> List[str]:
    """Build ``ingest`` arguments without a static cadence selection.

    The archive request retains the official available SPOC products. Scientific
    cadence is derived later from each candidate-owned FITS product.
    """
    return [
        "ingest",
        candidate_id,
        "--sectors",
        *[str(value) for value in sectors],
        "--products",
        products,
    ]


def build_detrend_argv(candidate_id: str, method: str, window_days: float) -> List[str]:
    """Build ``detrend`` arguments from wizard answers."""
    return [
        "detrend",
        candidate_id,
        "--method",
        method,
        "--window-days",
        repr(float(window_days)),
    ]


def build_search_argv(
    candidate_id: str,
    engine: str,
    period_min: float,
    period_max: float,
    detrending_method: Optional[str] = None,
) -> List[str]:
    """Build ``search`` arguments, threading the chosen detrending product."""
    argv = [
        "search",
        candidate_id,
        "--engine",
        engine,
        "--period-min",
        repr(float(period_min)),
        "--period-max",
        repr(float(period_max)),
    ]
    if detrending_method:
        argv.extend(["--detrending-method", detrending_method])
    return argv


def build_fit_argv(candidate_id: str, n_samples: int, eccentric: bool) -> List[str]:
    """Build ``fit`` arguments from wizard answers."""
    argv = ["fit", candidate_id, "--n-samples", str(int(n_samples))]
    if eccentric:
        argv.append("--eccentric")
    return argv


def build_vet_argv(candidate_id: str, n_draws: int) -> List[str]:
    """Build ``vet`` arguments from wizard answers."""
    return ["vet", candidate_id, "--n-draws", str(int(n_draws))]


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def parse_sectors(text: str) -> List[int]:
    """Parse a space-separated sector list into strictly positive integers."""
    values = [part for part in text.replace(",", " ").split() if part]
    sectors = [int(part) for part in values]
    if not sectors or any(value <= 0 or value > 9999 for value in sectors):
        raise ValueError("sectors must be one or more integers in 1..9999")
    if len(set(sectors)) != len(sectors):
        raise ValueError("duplicate sectors are not allowed")
    return sectors


def validate_period_bounds(period_min: float, period_max: float) -> Tuple[float, float]:
    """Enforce 0 < min < max for blind-search period bounds."""
    period_min = float(period_min)
    period_max = float(period_max)
    if not (period_min > 0.0 and period_max > period_min and period_max <= 3650.0):
        raise ValueError("period bounds must satisfy 0 < Pmin < Pmax <= 3650 days")
    return period_min, period_max


def validate_window_days(window_days: float) -> float:
    """Enforce a positive detrending window."""
    window_days = float(window_days)
    if not (window_days > 0.0 and window_days <= 60.0):
        raise ValueError("detrending window must lie in (0, 60] days")
    return window_days


# ---------------------------------------------------------------------------
# Prompt adapters -- thin wrappers so tests can script answers deterministically.
# ---------------------------------------------------------------------------


def _ask_text(console: Any, message: str, default: str = "") -> str:
    from rich.prompt import Prompt

    return Prompt.ask(message, default=default, console=console)


def _ask_int(console: Any, message: str, default: int) -> int:
    from rich.prompt import IntPrompt

    return IntPrompt.ask(message, default=default, console=console)


def _ask_float(console: Any, message: str, default: float) -> float:
    from rich.prompt import FloatPrompt

    return FloatPrompt.ask(message, default=default, console=console)


def _ask_choice(console: Any, message: str, choices: Sequence[Any], default: Any) -> Any:
    from rich.prompt import Prompt

    rendered = [str(choice) for choice in choices]
    answer = Prompt.ask(message, choices=rendered, default=str(default), console=console)
    for choice in choices:
        if str(choice) == answer:
            return choice
    return answer


def _ask_confirm(console: Any, message: str, default: bool) -> bool:
    from rich.prompt import Confirm

    return Confirm.ask(message, default=default, console=console)


# ---------------------------------------------------------------------------
# Wizard flow.
# ---------------------------------------------------------------------------


def _show_plan(console: Any, title: str, rows: Sequence[Tuple[str, str]]) -> bool:
    """Render a parameter summary panel and ask to execute."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    for key, value in rows:
        table.add_row(key, value)
    console.print(Panel(table, title=title, border_style="cyan"))
    return _ask_confirm(console, "Execute this step?", True)


def run_wizard(
    repository_root: Any,
    candidate_id: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> int:
    """Run the guided pipeline conversation and execute confirmed steps.

    Args:
        repository_root: Repository root forwarded to every executed command.
        candidate_id: Optional existing candidate; when omitted, Step 1 provisions.
        interactive: Override the terminal capability probe (for tests and
            embedded callers). ``None`` probes stdin/stdout TTY state.

    Returns:
        ``0`` when every executed step succeeded, ``1`` when any executed step
        failed, and ``2`` when no interactive terminal is available.
    """
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        print(
            "error: 'exonym wizard' requires an interactive terminal; "
            "use the individual commands instead.",
            file=sys.stderr,
        )
        return 2

    from rich.console import Console

    console = Console()
    root_argv = ["--root", str(repository_root)]

    def _execute(step_name: str, argv: List[str]) -> bool:
        from .__main__ import main

        console.print("[dim]$ exonym {0}[/dim]".format(" ".join(argv)))
        try:
            code = int(main(root_argv + argv))
        except SystemExit as exc:  # argparse errors surface as SystemExit(2)
            code = int(exc.code or 1)
        status = "[green]ok[/green]" if code == 0 else "[red]failed ({0})[/red]".format(code)
        console.print("  {0}: {1}".format(step_name, status))
        return code == 0

    failures = 0

    # ---- Step 1: target identity & provisioning ---------------------------
    if candidate_id is None:
        raw_name = _ask_text(console, "Target name (e.g. my-target-01)")
        candidate_id = raw_name.strip().lower()
        mission = str(_ask_choice(console, "Originating mission", MISSIONS, "tess"))
        tic = _ask_text(console, "TIC identifier (blank to skip)", default="").strip() or None
        tag_text = _ask_text(console, "Tags (comma-separated, blank to skip)", default="")
        tags = [tag.strip() for tag in tag_text.split(",") if tag.strip()]
        rows = [
            ("candidate id", candidate_id),
            ("mission", mission),
            ("tic", tic or "-"),
            ("tags", ", ".join(tags) if tags else "-"),
        ]
        if _show_plan(console, "Step 1 :: Provision candidate workspace", rows):
            if not _execute("init", build_init_argv(candidate_id, mission, tic, tags)):
                failures += 1

    # ---- Step 2: photometry ingestion -------------------------------------
    sector_text = _ask_text(console, "Observed sectors (space-separated integers)")
    sectors = parse_sectors(sector_text)
    products = str(_ask_choice(console, "Product type", PRODUCT_CHOICES, "lc"))
    if _show_plan(
        console,
        "Step 2 :: Ingest SPOC photometry",
        [
            ("sectors", " ".join(str(s) for s in sectors)),
            ("cadence", "derived from downloaded FITS metadata"),
            ("products", products),
        ],
    ):
        if not _execute(
            "ingest", build_ingest_argv(candidate_id, sectors, products)
        ):
            failures += 1

    # ---- Step 3: detrending -------------------------------------------------
    method_index = int(
        _ask_choice(
            console,
            "Detrending method",
            tuple(str(i + 1) for i in range(len(DETREND_METHODS))),
            "1",
        )
    )
    detrend_method = DETREND_METHODS[method_index - 1][0]
    window_days = validate_window_days(
        _ask_float(console, "Filter window (days)", _DEFAULT_WINDOW_DAYS)
    )
    if not _show_plan(
        console,
        "Step 3 :: Detrend light curve",
        [("method", detrend_method), ("window_days", repr(window_days))],
    ):
        detrend_method_for_search = None
    else:
        if not _execute(
            "detrend", build_detrend_argv(candidate_id, detrend_method, window_days)
        ):
            failures += 1
        detrend_method_for_search = detrend_method

    # ---- Step 4: transit search + stellar diagnostics ----------------------
    engine = str(_ask_choice(console, "Search engine", SEARCH_ENGINES, "bls"))
    period_min, period_max = validate_period_bounds(
        _ask_float(console, "Minimum period (days)", _DEFAULT_PERIOD_MIN),
        _ask_float(console, "Maximum period (days)", _DEFAULT_PERIOD_MAX),
    )
    search_rows = [
        ("engine", engine),
        ("period_min_d", repr(period_min)),
        ("period_max_d", repr(period_max)),
        ("detrending", detrend_method_for_search or "-"),
    ]
    if _show_plan(console, "Step 4 :: Transit search", search_rows):
        if not _execute(
            "search",
            build_search_argv(
                candidate_id, engine, period_min, period_max, detrend_method_for_search
            ),
        ):
            failures += 1
    if _ask_confirm(console, "Query archival Gaia/ExoFOP context now?", True):
        if not _execute("archive", ["archive", candidate_id]):
            failures += 1
    if _ask_confirm(console, "Fit stellar SED now?", True):
        if not _execute("sed", ["sed", candidate_id]):
            failures += 1

    # ---- Step 5: fitting & vetting ------------------------------------------
    fit_samples = int(_ask_int(console, "MCMC production samples", _DEFAULT_FIT_SAMPLES))
    eccentric = _ask_confirm(console, "Sample free eccentricity? (slower)", False)
    if fit_samples <= 0:
        raise ValueError("MCMC samples must be positive")
    if _show_plan(
        console,
        "Step 5 :: MCMC transit fit",
        [("samples", str(fit_samples)), ("eccentric", str(eccentric))],
    ):
        if not _execute(
            "fit", build_fit_argv(candidate_id, fit_samples, eccentric)
        ):
            failures += 1
    if _ask_confirm(console, "Run PRF localization first? (requires TPFs)", False):
        if not _execute("localization", ["localization", candidate_id]):
            failures += 1
    if _ask_confirm(console, "Run dilution sensitivity now?", False):
        if not _execute("dilution", ["dilution", candidate_id]):
            failures += 1
    vet_draws = int(_ask_int(console, "TRICERATOPS Monte Carlo draws", _DEFAULT_VET_DRAWS))
    if vet_draws <= 0:
        raise ValueError("Monte Carlo draws must be positive")
    if _show_plan(
        console,
        "Step 5b :: Statistical vetting",
        [("n_draws", str(vet_draws))],
    ):
        if not _execute("vet", build_vet_argv(candidate_id, vet_draws)):
            failures += 1

    console.rule("[bold]Wizard complete")
    return 0 if failures == 0 else 1
