"""Fetch transit priors from ExoFOP for a candidate workspace."""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from .workspace import CandidateWorkspace


def fetch_exofop_priors(workspace: CandidateWorkspace) -> List[Path]:
    """Fetch TOI transit parameters from ExoFOP and save to config/signals/.

    Returns a list of created JSON configuration file paths.
    """
    tic_id = workspace.metadata.get("identifiers", {}).get("tic")
    if not tic_id:
        raise ValueError("candidate lacks a TIC identifier")

    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    req = urllib.request.Request(url, headers={"User-Agent": "exonym/1.0.0"})

    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"ExoFOP returned status {response.status}")
        raw_csv = response.read().decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(raw_csv))
    matches = []
    for row in reader:
        try:
            if str(row.get("TIC ID", "")).strip() == str(tic_id).strip():
                matches.append(row)
        except Exception:
            continue

    if not matches:
        return []

    signals_dir = workspace.path / "config" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []

    for match in matches:
        toi = match.get("TOI", "")
        if "." in toi:
            signal_suffix = "." + toi.split(".")[1]
        else:
            signal_suffix = ".01"

        try:
            period = float(match.get("Period (days)", 0.0))
            epoch = float(match.get("Epoch (BJD)", 0.0))
            depth = float(match.get("Depth (ppm)", 0.0))
            duration = float(match.get("Duration (hours)", 0.0))
        except ValueError:
            continue

        if period <= 0.0:
            continue

        # The BJD in ExoFOP is usually BJD_TDB. We subtract 2457000 to get BTJD.
        if epoch > 2450000:
            epoch_btjd = epoch - 2457000.0
        else:
            epoch_btjd = epoch

        config = {
            "transit": {
                "period_days": round(period, 6),
                "epoch_btjd": round(epoch_btjd, 5),
                "depth_ppm": round(depth, 2),
                "duration_hours": round(duration, 2),
                "source": "nasa-exofop"
            }
        }

        config_path = signals_dir / f"transit_config{signal_suffix}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written_paths.append(config_path)

    return written_paths
