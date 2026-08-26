"""Unit coverage for the sticky telemetry HUD and its plain-text fallback."""

import io
from unittest.mock import patch

import pytest

from exonym.telemetry import LiveTelemetry, _current_rss_bytes, _format_bytes, _format_elapsed


def test_format_helpers():
    assert _format_bytes(None) == "-"
    assert _format_bytes(1536) == "1.50 KB"
    assert _format_elapsed(0) == "00:00:00"
    assert _format_elapsed(3723) == "01:02:03"


def test_current_rss_helper_returns_int_or_none():
    value = _current_rss_bytes()
    assert value is None or (isinstance(value, int) and value > 0)


def test_plain_mode_writes_no_ansi_and_reports_completion(capsys):
    with LiveTelemetry(
        "hud-target", phase="analysis", step_name="MCMC fit", interactive=False
    ) as hud:
        assert hud.plain is True
        hud.report_progress(5, 10)
        hud.field("walkers", 50)
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert "MCMC fit started" in captured.out
    assert "finished in" in captured.out


def test_live_mode_drives_progress_and_never_raises(monkeypatch):
    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.updates = []

        def add_task(self, description, total=None):
            return "task"

        def update(self, task_id, **kwargs):
            self.updates.append(kwargs)

        def reset(self, task_id, **kwargs):
            self.updates.append({"reset": kwargs})

    class FakeLive:
        instances = []

        def __init__(self, renderable, refresh_per_second=4, transient=True):
            self.renderable = renderable
            FakeLive.instances.append(self)
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.entered = False

        def update(self, renderable):
            self.renderable = renderable

    import rich.live as rich_live
    import rich.progress as rich_progress

    progress_proxy = FakeProgress()
    monkeypatch.setattr(rich_progress, "Progress", lambda *a, **k: progress_proxy)
    monkeypatch.setattr(rich_live, "Live", FakeLive)
    for name in ("BarColumn", "TextColumn", "TimeElapsedColumn"):
        monkeypatch.setattr(
            rich_progress, name, lambda *a, **k: name, raising=False
        )

    with LiveTelemetry(
        "hud-target", phase="vetting", step_name="BLS", interactive=True
    ) as hud:
        hud.report_progress(40, 100)
        hud.set_step("screening")
        hud.note_text("grid=uniform")

    assert FakeLive.instances and FakeLive.instances[0].entered is False
    assert any(kwargs.get("completed") == 40 for kwargs in progress_proxy.updates)
