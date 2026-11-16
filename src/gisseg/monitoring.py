"""Input monitoring for the segmentation service, with a retrain trigger.

Scanners, protocols and patient populations shift. Three checks compare incoming volumes with the
training reference:
  intensity drift        PSI of per-slice mean intensity (after the same normalisation)
  unseen resolutions     share of incoming volumes whose image size never appeared in training
  organ prevalence       change in the share of slices that contain any organ (needs labels, e.g. audits)
Any alarm sets ``retrain_recommended``.

    uv run python -m gisseg.monitoring      # reference = training folds, incoming = validation fold
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gisseg.config import Config, load_config
from gisseg.data.dataset import folds

THRESHOLDS = {"intensity_psi": 0.2, "unseen_resolution_share": 0.2, "prevalence_change": 0.15}


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / reference.size, 1e-6, None)
    cur = np.clip(np.histogram(current, edges)[0] / current.size, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def slice_means(cache: Path, case_days: list[str]) -> np.ndarray:
    return np.concatenate(
        [np.load(cache / f"{cd}_img.npy", mmap_mode="r").mean(axis=(1, 2)) for cd in case_days]
    )


def check(cfg: Config, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    ref_sizes = set(zip(reference["height"], reference["width"], strict=True))
    unseen = float(
        np.mean([(h, w) not in ref_sizes for h, w in zip(current["height"], current["width"], strict=True)])
    )
    prev_ref = reference["positive_slices"].sum() / reference["n_slices"].sum()
    prev_cur = current["positive_slices"].sum() / current["n_slices"].sum()
    values = {
        "intensity_psi": psi(
            slice_means(cfg.paths.cache, reference["case_day"].tolist()),
            slice_means(cfg.paths.cache, current["case_day"].tolist()),
        ),
        "unseen_resolution_share": unseen,
        "prevalence_change": float(abs(prev_cur - prev_ref)),
    }
    alerts = {k: v > THRESHOLDS[k] for k, v in values.items()}
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_volumes": len(reference),
        "current_volumes": len(current),
        "values": values,
        "alerts": alerts,
        "retrain_recommended": any(alerts.values()),
    }


def to_markdown(r: dict[str, Any]) -> str:
    L = [
        "# Monitoring report",
        "",
        f"Checked {r['checked_at']}: {r['current_volumes']} incoming volumes "
        f"vs {r['reference_volumes']} reference volumes.",
        "",
        "| Check | Value | Alert if > | Alert |",
        "|---|---:|---:|---|",
    ]
    for k, v in r["values"].items():
        L.append(
            f"| {k.replace('_', ' ')} | {v:.4f} | {THRESHOLDS[k]} | {'yes' if r['alerts'][k] else 'no'} |"
        )
    L += ["", f"**Retrain recommended: {'yes' if r['retrain_recommended'] else 'no'}**"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    index = pd.read_csv(cfg.paths.cache / "index.csv")
    index["fold"] = folds(index, cfg.folds)
    r = check(cfg, index[index["fold"] != cfg.valid_fold], index[index["fold"] == cfg.valid_fold])
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "monitoring.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (cfg.paths.reports / "monitoring.md").write_text(to_markdown(r), encoding="utf-8")
    print(to_markdown(r))


if __name__ == "__main__":
    main()
