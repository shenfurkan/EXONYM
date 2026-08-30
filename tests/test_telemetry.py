"""Unit coverage for the sticky telemetry HUD and its plain-text fallback."""

from exonym.telemetry import (
    LiveTelemetry,
    _current_rss_bytes,
    _format_bytes,
    _format_elapsed,
    _format_time,
    _progress_label,
)


def test_format_helpers():
    assert _format_bytes(None) == "-"
    assert _format_bytes(1536) == "1.50 KB"
    assert _format_elapsed(0) == "00:00:00"
    assert _format_elapsed(3723) == "01:02:03"
    assert _format_time(None) == "--:--:--"
    assert _format_time(2.0) == "00:00:02"
    assert _progress_label(0, None) == " -- / --"
    assert _progress_label(4, 10) == " 40% /  60% left"
    assert _progress_label(12, 10) == "100% /   0% left"


def test_current_rss_helper_returns_int_or_none():
    value = _current_rss_bytes()
    assert value is None or (isinstance(value, int) and value > 0)


def test_plain_mode_writes_no_ansi_and_reports_progress(capsys, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("exonym.telemetry.time.monotonic", lambda: clock[0])
    with LiveTelemetry(
        "hud-target", phase="analysis", step_name="MCMC fit", interactive=False
    ) as hud:
        assert hud.plain is True
        hud.report_progress(0, 10)
        clock[0] = 2.0
        hud.report_progress(4, 10)
        clock[0] = 6.0
        hud.report_progress(5, 10, sub_phase="production")
        hud.set_step("writing output", note="serializing")
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert "MCMC fit started" in captured.out
    assert "production" in captured.out
    assert "writing output" in captured.out
    assert "finished in" in captured.out


def test_live_mode_drives_progress_and_never_raises(monkeypatch):
    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.updates = []

        def add_task(self, description, total=None, **fields):
            return "task"

        def update(self, task_id, **kwargs):
            self.updates.append(kwargs)

        def reset(self, task_id, **kwargs):
            self.updates.append({"reset": kwargs})

    class FakeLive:
        instances = []

        def __init__(self, renderable, **kwargs):
            self.renderable = renderable
            self.kwargs = kwargs
            FakeLive.instances.append(self)
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.entered = False

        def update(self, renderable, refresh=False):
            self.renderable = renderable
            self.refresh = refresh

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
        hud.set_step("screening", note="grid=uniform")

    assert FakeLive.instances and FakeLive.instances[0].entered is False
    assert any(kwargs.get("completed") == 40 for kwargs in progress_proxy.updates)
    assert any(kwargs.get("progress_label") == " 40% /  60% left" for kwargs in progress_proxy.updates)
    assert FakeLive.instances[0].kwargs == {
        "refresh_per_second": 6,
        "transient": True,
        "redirect_stdout": True,
        "redirect_stderr": True,
        "vertical_overflow": "crop",
        "get_renderable": hud._refresh_renderer,
    }
    assert FakeLive.instances[0].refresh is False


def test_eta_mcmc_evidence_and_step_timer_reset(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("exonym.telemetry.time.monotonic", lambda: clock[0])
    hud = LiveTelemetry("hud-target", phase="analysis", step_name="MCMC fit", interactive=False)

    hud.report_progress(0, 20)
    clock[0] = 2.0
    hud.report_progress(4, 20)
    assert hud._estimate_remaining_seconds() == 8.0

    hud.report_mcmc(
        6,
        20,
        burn_in=10,
        production=10,
        acceptance_fraction=0.25,
        autocorr_tau=3.5,
    )
    assert hud.sub_phase == "burn-in"
    assert hud.fields["phase"] == "6/10"
    assert hud.fields["acceptance"] == "0.250"
    assert hud.fields["tau"] == "3.5"

    hud.report_evidence(8, log_z=12.5, log_z_error=0.2, likelihood_calls=40)
    assert hud.total is None
    assert hud.sub_phase == "nested sampling"
    assert hud.fields["ln Z"] == "12.500"

    clock[0] = 9.0
    hud.set_step("writing output")
    assert hud._start == 9.0
    assert hud._rate_per_second is None
