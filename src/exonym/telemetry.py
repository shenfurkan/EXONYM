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

import sys
import time
from typing import Any, Dict, Optional

_HUD_HEIGHT = 4


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
        self.note: Optional[str] = None
        self.fields: Dict[str, str] = {}
        self.done = 0
        self.total = total_steps
        self._start = time.monotonic()
        self._live: Any = None
        self._task_id: Any = None
        self._progress: Any = None
        if interactive is None:
            interactive = sys.stdout.isatty()
        self.plain = not interactive

    @staticmethod
    def _resolve_phase(repository_root: Any) -> str:
        """Best-effort workflow-phase lookup from the candidate metadata."""
        return "unknown"

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
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
        from rich.text import Text

        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=28),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        )
        self._task_id = self._progress.add_task(
            self.step_name, total=self.total if self.total else None
        )

        def _render() -> Group:
            info = Text(
                "Target: {0}  |  Phase: {1}  |  Step: {2}".format(
                    self.candidate_id, self.phase.upper(), self.step_name
                ),
                style="bold white",
            )
            stats = Text(
                "Memory: {0}  |  Elapsed: {1}{2}".format(
                    _format_bytes(_current_rss_bytes()),
                    _format_elapsed(time.monotonic() - self._start),
                    "  |  {0}".format(self.note) if self.note else "",
                )
                + (
                    "  |  "
                    + "  ".join("{0}: {1}".format(k, v) for k, v in self.fields.items())
                    if self.fields
                    else ""
                ),
                style="dim",
            )
            return Panel(
                Group(info, self._progress, stats),
                title="[bold white]EXONYM ENGINE TELEMETRY[/]",
                border_style="dim white",
            )

        self._live = Live(_render(), refresh_per_second=4, transient=True)
        self._live.__enter__()
        self._refresh_renderer = _render  # type: ignore[attr-defined]
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

    def _redraw(self) -> None:
        """Push a refreshed renderable into the live region, ignoring errors."""
        if self._live is None:
            return
        try:
            update = getattr(self._live, "update", None)
            renderer = getattr(self, "_refresh_renderer", None)
            if update is not None and renderer is not None:
                update(renderer())
        except Exception:  # noqa: BLE001
            pass

    def report_progress(self, done: int, total: int) -> None:
        """Adopt sampler-reported iteration counts (never raises)."""
        try:
            done = max(0, int(done))
            total = max(done, int(total))
            self.done, self.total = done, total
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, completed=done, total=total)
            self._redraw()
        except Exception:  # noqa: BLE001
            pass

    def set_step(self, step_name: str, total: Optional[int] = None) -> None:
        """Switch the displayed step label."""
        try:
            self.step_name = step_name
            if total is not None:
                self.total = total
            if self._progress is not None and self._task_id is not None:
                self._progress.reset(
                    self._task_id,
                    description=step_name,
                    total=total if total else None,
                    completed=0,
                )
            self._redraw()
        except Exception:  # noqa: BLE001
            pass

    def note_text(self, text: str) -> None:
        """Set the free-form trailing note shown beside memory/elapsed."""
        self.note = text
        self._redraw()

    def field(self, key: str, value: Any) -> None:
        """Upsert one key/value indicator (e.g. walkers, acceptance)."""
        self.fields[str(key)] = str(value)
        self._redraw()
