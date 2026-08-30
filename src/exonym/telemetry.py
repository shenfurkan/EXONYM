"""Sticky live status and progress telemetry for long-running analysis.

``LiveTelemetry`` renders an anchored bottom HUD (via ``rich.live.Live``)
while heavy commands run, and degrades to plain single-line logging when
stdout is not a TTY so redirected output, CI logs, and ``cmd.exe`` pipes stay
ANSI-free.  Every mutator is exception-safe by contract: telemetry must never
abort a scientific run.

Scientific boundary
-------------------
The HUD is presentation-only.  It reports sampler iteration counts supplied
by the caller and never interprets them as convergence evidence.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, Optional


_PLAIN_PROGRESS_INTERVAL_SECONDS = 5.0
_RATE_EMA_ALPHA = 0.25


def _format_bytes(num_bytes: Optional[int]) -> str:
    """Render a byte count as a compact human-readable string."""
    if num_bytes is None:
        return "-"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return "{0:.2f} {1}".format(value, unit)
        value /= 1024.0
    return "{0:.2f} GB".format(value)


def _format_elapsed(seconds: float) -> str:
    """Render elapsed seconds as HH:MM:SS."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return "{0:02d}:{1:02d}:{2:02d}".format(hours, minutes, secs)


def _format_time(seconds: Optional[float]) -> str:
    """Render a finite duration, or an unavailable marker."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "--:--:--"
    return _format_elapsed(seconds)


def _progress_label(done: int, total: Optional[int]) -> str:
    """Render completed and remaining percentages only for known totals."""
    if total is None or total <= 0:
        return " -- / --"
    completed = 100.0 * min(max(0, done), total) / total
    return "{0:>3.0f}% / {1:>3.0f}% left".format(completed, 100.0 - completed)


def _current_rss_bytes() -> Optional[int]:
    """Best-effort resident set size of the current process in bytes."""
    try:
        if sys.platform == "win32":  # pragma: no cover - exercised via ctypes on Windows only
            import ctypes.wintypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if not ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return None
            return int(counters.WorkingSetSize)
        import resource

        maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return int(maximum_rss)
        return int(maximum_rss) * 1024
    except Exception:  # noqa: BLE001
        return None


class LiveTelemetry:
    """Context manager rendering a sticky bottom status panel.

    Args:
        candidate_id: Candidate identifier shown in the header.
        repository_root: Optional root used to resolve the workflow phase
            from the candidate metadata; failures fall back to "unknown".
        phase: Explicit phase label overriding metadata resolution.
        step_name: Initial step label.
        total_steps: Optional total for determinate progress.
    """

    def __init__(
        self,
        candidate_id: str,
        repository_root: Any = None,
        phase: Optional[str] = None,
        step_name: str = "working",
        total_steps: Optional[int] = None,
        interactive: Optional[bool] = None,
    ) -> None:
        self.candidate_id = candidate_id
        self.phase = phase or self._resolve_phase_for(repository_root)
        self.step_name = step_name
        self.sub_phase: Optional[str] = None
        self.note: Optional[str] = None
        self.fields: Dict[str, str] = {}
        self.done = 0
        self.total = total_steps
        self._start = time.monotonic()
        self._last_rate_time: Optional[float] = None
        self._last_rate_done: Optional[int] = None
        self._rate_per_second: Optional[float] = None
        self._last_plain_progress_at = self._start
        self._live: Any = None
        self._task_id: Any = None
        self._progress: Any = None
        if interactive is None:
            interactive = sys.stdout.isatty()
        self.plain = not interactive

    def _resolve_phase_for(self, repository_root: Any) -> str:
        """Resolve the workflow phase using this instance's candidate id."""
        if repository_root is None:
            return "unknown"
        try:
            from .workspace import load_candidate

            workspace = load_candidate(repository_root, self.candidate_id)
            return str(workspace.metadata["workflow"]["phase"])
        except Exception:  # noqa: BLE001
            return "unknown"

    # -- context management -------------------------------------------------

    def __enter__(self) -> "LiveTelemetry":
        if self.plain:
            print("[exonym] {0} :: {1} started".format(self.phase, self.step_name), flush=True)
            return self
        try:
            from rich.console import Group
            from rich.live import Live
            from rich.panel import Panel
            from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
            from rich.text import Text

            self._progress = Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=28),
                TextColumn("[progress.percentage]{task.fields[progress_label]}"),
                TimeElapsedColumn(),
            )
            self._task_id = self._progress.add_task(
                self.step_name,
                total=self.total if self.total else None,
                progress_label=_progress_label(self.done, self.total),
            )

            def _render() -> Group:
                info = Text(
                    "Target: {0}  |  Phase: {1}  |  Step: {2}{3}".format(
                        self.candidate_id,
                        self.phase.upper(),
                        self.step_name,
                        "  |  Sub-phase: {0}".format(self.sub_phase) if self.sub_phase else "",
                    ),
                    style="bold white",
                    no_wrap=True,
                    overflow="ellipsis",
                )
                eta = self._estimate_remaining_seconds()
                stats = Text(
                    "Memory: {0}  |  Elapsed: {1}  |  ETA: {2}{3}".format(
                        _format_bytes(_current_rss_bytes()),
                        _format_elapsed(time.monotonic() - self._start),
                        _format_time(eta),
                        "  |  {0}".format(self.note) if self.note else "",
                    )
                    + (
                        "  |  "
                        + "  ".join("{0}: {1}".format(k, v) for k, v in self.fields.items())
                        if self.fields
                        else ""
                    ),
                    style="dim",
                    no_wrap=True,
                    overflow="ellipsis",
                )
                return Panel(
                    Group(info, self._progress, stats),
                    title="[bold white]EXONYM ENGINE TELEMETRY[/]",
                    border_style="dim white",
                )

            self._refresh_renderer = _render  # type: ignore[attr-defined]
            self._live = Live(
                _render(),
                refresh_per_second=6,
                transient=True,
                redirect_stdout=True,
                redirect_stderr=True,
                vertical_overflow="crop",
                get_renderable=_render,
            )
            self._live.__enter__()
        except Exception:  # noqa: BLE001 - presentation must not abort a run.
            self._live = None
            self._progress = None
            self._task_id = None
            self.plain = True
            print("[exonym] {0} :: {1} started".format(self.phase, self.step_name), flush=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = _format_elapsed(time.monotonic() - self._start)
        if self._live is not None:
            try:
                self._live.__exit__(exc_type, exc_val, exc_tb)
            except Exception:  # noqa: BLE001
                pass
            self._live = None
        status = "failed" if exc_type else "finished"
        print(
            "[exonym] {0} :: {1} {2} in {3}".format(
                self.phase, self.step_name, status, elapsed
            ),
            flush=True,
        )

    # -- updates -------------------------------------------------------------

    def _sync_render(self) -> None:
        """Queue an updated renderable without competing with Live's refresh loop."""
        if self._live is None:
            return
        try:
            update = getattr(self._live, "update", None)
            renderer = getattr(self, "_refresh_renderer", None)
            if update is not None and renderer is not None:
                update(renderer(), refresh=False)
        except Exception:  # noqa: BLE001
            pass

    def _reset_rate(self) -> None:
        self._last_rate_time = None
        self._last_rate_done = None
        self._rate_per_second = None

    def _record_rate(self, now: float, done: int, total: Optional[int]) -> None:
        if total is None or total <= 0:
            self._reset_rate()
            return
        if self._last_rate_time is not None and self._last_rate_done is not None:
            elapsed = now - self._last_rate_time
            delta = done - self._last_rate_done
            if elapsed > 0.0 and delta > 0:
                sample = float(delta) / elapsed
                self._rate_per_second = (
                    sample
                    if self._rate_per_second is None
                    else _RATE_EMA_ALPHA * sample + (1.0 - _RATE_EMA_ALPHA) * self._rate_per_second
                )
        self._last_rate_time = now
        self._last_rate_done = done

    def _estimate_remaining_seconds(self) -> Optional[float]:
        if self.total is None or self._rate_per_second is None or self._rate_per_second <= 0.0:
            return None
        return float(max(0, self.total - self.done)) / self._rate_per_second

    def _plain_progress(self, force: bool = False) -> None:
        if not self.plain:
            return
        now = time.monotonic()
        if not force and now - self._last_plain_progress_at < _PLAIN_PROGRESS_INTERVAL_SECONDS:
            return
        self._last_plain_progress_at = now
        print(
            "[exonym] {0} :: {1}{2} {3} ETA {4}{5}".format(
                self.phase,
                self.step_name,
                " ({0})".format(self.sub_phase) if self.sub_phase else "",
                _progress_label(self.done, self.total),
                _format_time(self._estimate_remaining_seconds()),
                " :: {0}".format(self.note) if self.note else "",
            ),
            flush=True,
        )

    def report_progress(
        self,
        done: int,
        total: Optional[int] = None,
        sub_phase: Optional[str] = None,
        **fields: Any,
    ) -> None:
        """Adopt engine progress, phase, and optional metrics without raising."""
        try:
            done = max(0, int(done))
            total = max(done, int(total)) if total is not None else None
            self.done, self.total = done, total
            self.sub_phase = str(sub_phase) if sub_phase else None
            for key, value in fields.items():
                if value is not None:
                    self.fields[str(key)] = str(value)
            self._record_rate(time.monotonic(), done, total)
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=done,
                    total=total,
                    progress_label=_progress_label(done, total),
                )
            self._plain_progress()
            self._sync_render()
        except Exception:  # noqa: BLE001
            pass

    def report_mcmc(
        self,
        done: int,
        total: int,
        burn_in: int,
        production: int,
        acceptance_fraction: Optional[float] = None,
        autocorr_tau: Optional[float] = None,
        **_ignored: Any,
    ) -> None:
        """Report globally counted emcee progress with its current sampling phase."""
        phase = "burn-in" if done < burn_in else "production"
        phase_done = done if phase == "burn-in" else max(0, done - burn_in)
        phase_total = burn_in if phase == "burn-in" else production
        fields: Dict[str, Any] = {"phase": "{0}/{1}".format(phase_done, phase_total)}
        if acceptance_fraction is not None and math.isfinite(float(acceptance_fraction)):
            fields["acceptance"] = "{0:.3f}".format(float(acceptance_fraction))
        if autocorr_tau is not None and math.isfinite(float(autocorr_tau)):
            fields["tau"] = "{0:.1f}".format(float(autocorr_tau))
        self.report_progress(done, total, sub_phase=phase, **fields)

    def report_evidence(
        self,
        iteration: int,
        log_z: Optional[float] = None,
        log_z_error: Optional[float] = None,
        likelihood_calls: Optional[int] = None,
        **_ignored: Any,
    ) -> None:
        """Report an indeterminate nested-sampling evidence update."""
        fields: Dict[str, Any] = {}
        if log_z is not None and math.isfinite(float(log_z)):
            fields["ln Z"] = "{0:.3f}".format(float(log_z))
        if log_z_error is not None and math.isfinite(float(log_z_error)):
            fields["ln Z err"] = "{0:.3f}".format(float(log_z_error))
        if likelihood_calls is not None:
            fields["calls"] = likelihood_calls
        self.report_progress(iteration, None, sub_phase="nested sampling", **fields)

    def set_step(
        self,
        step_name: str,
        total: Optional[int] = None,
        note: Optional[str] = None,
        sub_phase: Optional[str] = None,
    ) -> None:
        """Switch the displayed step and coalesce its optional status text."""
        try:
            total = int(total) if total is not None else None
            changed = self.step_name != step_name or self.total != total
            self.note = note if note is not None else self.note
            self.sub_phase = sub_phase
            if not changed:
                self._plain_progress(force=note is not None)
                self._sync_render()
                return
            self.step_name = step_name
            self.done, self.total = 0, total
            self._start = time.monotonic()
            self._reset_rate()
            if self._progress is not None and self._task_id is not None:
                self._progress.reset(
                    self._task_id,
                    description=step_name,
                    total=total if total else None,
                    completed=0,
                    progress_label=_progress_label(0, total),
                )
            self._plain_progress(force=True)
            self._sync_render()
        except Exception:  # noqa: BLE001
            pass

    def note_text(self, text: str) -> None:
        """Set the free-form trailing note shown beside memory/elapsed."""
        try:
            self.note = str(text)
            self._plain_progress()
            self._sync_render()
        except Exception:  # noqa: BLE001
            pass

    def field(self, key: str, value: Any) -> None:
        """Upsert one key/value indicator (e.g. walkers, acceptance)."""
        try:
            self.fields[str(key)] = str(value)
            self._plain_progress()
            self._sync_render()
        except Exception:  # noqa: BLE001
            pass
