"""Tests for candidate-local manuscript export."""

import json

from exonym.paper_export import export_paper
from exonym.workspace import create_candidate


def test_export_paper_uses_safe_macro_overrides_and_available_posteriors(tmp_path):
    workspace = create_candidate(tmp_path, "paper-synthetic")
    outputs = workspace.path / "outputs"
    (outputs / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "ephemeris": {"period_days": 3.5, "epoch_btjd": 1200.25},
                "posterior": {
                    "rp_rs": {"median": 0.1, "plus": 0.01, "minus": 0.01},
                    "a_rs": {"median": 12.0, "plus": 1.0, "minus": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "sed_fit_results.json").write_text(
        json.dumps(
            {
                "posterior": {
                    "teff_k": {"median": 5700.0, "plus": 75.0, "minus": 75.0},
                    "radius_solar": {"median": 1.0, "plus": 0.1, "minus": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )

    macro_path = export_paper(workspace)

    macros = macro_path.read_text(encoding="utf-8")
    assert r"\renewcommand{\ExonymRadiusRatio}{0.1^{+0.01}_{-0.01}}" in macros
    assert "$0.1" not in macros
    assert r"\renewcommand{\ExonymClaimEligibility}{false}" in macros
    assert (workspace.path / "paper" / "paper_template.tex").is_file()
    manifest = json.loads(
        (workspace.path / "paper" / "generated" / "paper_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["claim_eligible"] is False
    assert [source["path"] for source in manifest["sources"]] == [
        "outputs/mcmc_transit_fit.json",
        "outputs/sed_fit_results.json",
    ]
def test_paper_export_tiny_value_formatting():
    """_number() must not render small values as literal '0'."""
    from exonym.paper_export import _number

    tiny = _number(3.2e-7, digits=6)
    assert tiny is not None
    assert tiny != "0", "3.2e-7 must not render as '0'"
    assert "e" in tiny.lower() or tiny.replace(".", "").strip("0") != "", (
        "tiny value must retain nonzero digits: got {0!r}".format(tiny)
    )

    moderate = _number(0.00123, digits=6)
    assert moderate is not None
    assert "0.00123" in moderate, (
        "0.00123 must be preserved: got {0!r}".format(moderate)
    )


def test_paper_export_preserves_tiny_posterior_without_inventing_an_upper_limit():
    """Tiny posterior values must render as valid TeX, not a data-free limit."""
    from exonym.paper_export import _posterior_latex

    # This is a fractional-flux representation of 5 +/- 50 ppm.  The
    # posterior summary does not declare a nondetection or upper-limit
    # semantic, so the export must preserve its asymmetric-summary contract.
    rendered = _posterior_latex(
        {"median": 5.0e-6, "plus": 50.0e-6, "minus": 50.0e-6}, digits=6
    )

    assert rendered == r"{5 \times 10^{-6}}^{+5 \times 10^{-5}}_{-5 \times 10^{-5}}"
    assert "<" not in rendered
    assert "{0}" not in rendered


def test_export_paper_signal_scopes_figures_and_nonfinite_vetting_values(tmp_path):
    workspace = create_candidate(tmp_path, "paper-synthetic-signal")
    outputs = workspace.path / "outputs"
    (outputs / "triceratops_report.02.json").write_text(
        '{"FPP": NaN, "NFPP": Infinity}\n', encoding="utf-8"
    )
    figure = workspace.path / "figures" / "phase_folded_lc.02.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"synthetic image")

    macro_path = export_paper(workspace, signal=".02")

    macros = macro_path.read_text(encoding="utf-8")
    manifest = json.loads(
        (workspace.path / "paper" / "generated" / "paper_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert r"\renewcommand{\ExonymFpp}{Not available}" in macros
    assert r"\renewcommand{\ExonymNfpp}{Not available}" in macros
    assert r"\renewcommand{\ExonymFigureTransit}{../figures/phase_folded_lc.02.png}" in macros
    assert manifest["signal"] == ".02"
    assert {source["path"] for source in manifest["sources"]} == {
        "outputs/triceratops_report.02.json"
    }


def test_export_paper_degrades_when_vetting_report_is_absent(tmp_path):
    workspace = create_candidate(tmp_path, "paper-synthetic-no-vet")

    macro_path = export_paper(workspace)

    macros = macro_path.read_text(encoding="utf-8")
    manifest = json.loads(
        (workspace.path / "paper" / "generated" / "paper_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert r"\renewcommand{\ExonymFpp}{Not available}" in macros
    assert r"\renewcommand{\ExonymNfpp}{Not available}" in macros
    assert manifest["sources"] == []


def test_latex_text_escapes_all_special_characters_including_backslash_first():
    from exonym.paper_export import _latex_text

    assert _latex_text(r"\&%$#_{}~^") == (
        r"\textbackslash{}\&\%\$\#\_\{\}\textasciitilde{}\textasciicircum{}"
    )
