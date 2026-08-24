"""Export candidate-owned analysis evidence into a manuscript macro bundle.

The exporter collects already-written, candidate-local artifacts and formats
their declared values as TeX-safe macros together with an export manifest. It
preserves scientific notation for small finite values rather than rounding a
measurement into a different numerical statement.

Scientific boundary:
    The bundle is a convenience layer for manuscript drafting. It is explicitly
    claim-ineligible and cannot upgrade incomplete, uncalibrated, or otherwise
    unsuitable source evidence into a scientific conclusion.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .resources import iter_template_texts
from .workspace import CandidateWorkspace, validate_signal_suffix


_FIGURES = {
    "Transit": "figures/phase_folded_lc{suffix}.png",
    "Corner": "figures/corner_plot{suffix}.png",
    "Localization": "figures/localization.png",
    "Sed": "figures/sed_fit.png",
    "Fpp": "figures/triceratops_fpp.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latex_text(value: object) -> str:
    """Escape candidate-provided text before it reaches a TeX macro."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _number(value: object, digits: int = 5) -> Optional[str]:
    """Format one finite scalar for use inside an already-mathematical TeX macro.

    Nonzero values that Python emits with an ``e`` exponent are converted to
    TeX scientific notation.  This preserves a tiny posterior value rather
    than turning it into a rounded decimal zero, without assigning an
    upper-limit interpretation that is absent from the source artifact.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    rendered = f"{numeric:.{digits}g}"
    mantissa, exponent_separator, exponent = rendered.lower().partition("e")
    if not exponent_separator:
        return rendered
    return r"{0} \times 10^{{{1}}}".format(mantissa, int(exponent))


def _posterior_latex(summary: object, digits: int = 5) -> str:
    if not isinstance(summary, dict):
        return "Not available"
    median = _number(summary.get("median"), digits)
    plus = _number(summary.get("plus"), digits)
    minus = _number(summary.get("minus"), digits)
    if median is None:
        return "Not available"
    if plus is None or minus is None:
        return median
    # A scientific-notation median ends in ``10^{...}``.  Group it before
    # attaching the posterior interval so TeX does not see two superscripts.
    median_term = "{" + median + "}" if r"\times" in median else median
    return r"%s^{+%s}_{-%s}" % (median_term, plus, minus)


def _value_latex(value: object, digits: int = 5) -> str:
    number = _number(value, digits)
    return number if number is not None else "Not available"


def _candidate_template_text(repository_root: Path) -> str:
    for relative, text in iter_template_texts(repository_root):
        if relative.as_posix() == "paper/paper_template.tex":
            return text
    raise FileNotFoundError("paper template is unavailable from templates/paper/paper_template.tex")


def _ensure_candidate_template(workspace: CandidateWorkspace) -> Path:
    paper_dir = workspace.path / "paper"
    template_path = paper_dir / "paper_template.tex"
    if not template_path.is_file():
        paper_dir.mkdir(parents=True, exist_ok=True)
        template_path.write_text(_candidate_template_text(workspace.repository_root), encoding="utf-8")
    return template_path


def _first_json(outputs: Path, names: Iterable[str]) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    for name in names:
        path = outputs / name
        payload = _read_json(path)
        if payload is not None:
            return path, payload
    return None, None


def _macro_lines(
    workspace: CandidateWorkspace,
    fit: Optional[Dict[str, Any]],
    sed: Optional[Dict[str, Any]],
    vet: Optional[Dict[str, Any]],
    signal: Optional[str],
) -> List[str]:
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    posterior = fit.get("posterior", {}) if isinstance(fit, dict) else {}
    ephemeris = fit.get("ephemeris", {}) if isinstance(fit, dict) else {}
    sed_posterior = sed.get("posterior", {}) if isinstance(sed, dict) else {}
    fpp = vet.get("FPP") if isinstance(vet, dict) else None
    nfpp = vet.get("NFPP") if isinstance(vet, dict) else None
    lines = [
        "% Generated by exonym export-paper. Do not hand-edit this file.",
        r"\renewcommand{\ExonymCandidateId}{%s}" % _latex_text(workspace.candidate_id),
        r"\renewcommand{\ExonymExportedUtc}{%s}" % datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        r"\renewcommand{\ExonymPeriodDays}{%s}" % _value_latex(ephemeris.get("period_days"), 7),
        r"\renewcommand{\ExonymEpochBtjd}{%s}" % _value_latex(ephemeris.get("epoch_btjd"), 7),
        r"\renewcommand{\ExonymRadiusRatio}{%s}" % _posterior_latex(posterior.get("rp_rs"), 5),
        r"\renewcommand{\ExonymScaledSemimajorAxis}{%s}" % _posterior_latex(posterior.get("a_rs"), 4),
        r"\renewcommand{\ExonymInclinationDeg}{%s}" % _posterior_latex(posterior.get("inclination_deg"), 3),
        r"\renewcommand{\ExonymImpactParameter}{%s}" % _posterior_latex(posterior.get("impact_parameter"), 4),
        r"\renewcommand{\ExonymLimbDarkeningOne}{%s}" % _posterior_latex(posterior.get("u1"), 4),
        r"\renewcommand{\ExonymLimbDarkeningTwo}{%s}" % _posterior_latex(posterior.get("u2"), 4),
        r"\renewcommand{\ExonymStellarTeff}{%s}" % _posterior_latex(sed_posterior.get("teff_k"), 1),
        r"\renewcommand{\ExonymStellarRadius}{%s}" % _posterior_latex(sed_posterior.get("radius_solar"), 3),
        r"\renewcommand{\ExonymFpp}{%s}" % _value_latex(fpp, 6),
        r"\renewcommand{\ExonymNfpp}{%s}" % _value_latex(nfpp, 6),
        r"\renewcommand{\ExonymClaimEligibility}{false}",
    ]
    for label, pattern in _FIGURES.items():
        relative = pattern.format(suffix=suffix)
        lines.append(r"\renewcommand{\ExonymFigure%s}{../%s}" % (label, relative))
        lines.append(
            r"\providecommand{\ExonymFigure%sStatus}{%s}"
            % (label, "available" if (workspace.path / relative).is_file() else "missing")
        )
    return lines


def export_paper(workspace: CandidateWorkspace, signal: Optional[str] = None) -> Path:
    """Write manuscript macros and a source manifest from available candidate outputs.

    The exporter does not infer missing measurements, generate a claim, or
    compile a PDF. The generated TeX file is a candidate-local drafting aid.
    """
    signal = validate_signal_suffix(signal)
    _ensure_candidate_template(workspace)
    outputs = workspace.path / "outputs"
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    fit_path, fit = _first_json(outputs, ("mcmc_transit_fit{0}.json".format(suffix),))
    sed_path, sed = _first_json(outputs, ("sed_fit_results.json",))
    vet_path, vet = _first_json(
        outputs,
        ("triceratops_report{0}.json".format(suffix), "triceratops_results{0}.json".format(suffix)),
    )
    generated_dir = workspace.path / "paper" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    macro_path = generated_dir / "exonym_macros.tex"
    macro_path.write_text("\n".join(_macro_lines(workspace, fit, sed, vet, signal)) + "\n", encoding="utf-8")

    source_paths = [path for path in (fit_path, sed_path, vet_path) if path is not None]
    figures = []
    for pattern in _FIGURES.values():
        relative = pattern.format(suffix=suffix)
        path = workspace.path / relative
        if path.is_file():
            figures.append({"path": relative, "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "signal": signal,
        "template": "paper/paper_template.tex",
        "macros": "paper/generated/exonym_macros.tex",
        "sources": [
            {"path": path.relative_to(workspace.path).as_posix(), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "figures": figures,
        "claim_eligible": False,
        "caveat": "Exported values are drafting aids and preserve the source artifacts' scientific limitations.",
    }
    (generated_dir / "paper_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return macro_path
