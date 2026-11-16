"""Write RESULTS.md from the MLflow runs recorded in reports/metrics.json (exact provenance).

uv run python -m gisseg.report
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from gisseg.config import PROJECT_ROOT, Config, load_config, setup_mlflow


def build(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    setup_mlflow(cfg, root)
    meta = json.loads((cfg.paths.reports / "metrics.json").read_text(encoding="utf-8"))
    ids = meta["mlflow_run_ids"]
    model, base = mlflow.get_run(ids["model"]), mlflow.get_run(ids["baseline"])
    m, b, p = model.data.metrics, base.data.metrics, model.data.params
    L = [
        "# Results",
        "",
        f"Generated from MLflow on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m gisseg.report`.",
        "Metric: the competition score, **0.4 x Dice + 0.6 x (1 - normalised 3D Hausdorff)** "
        "(higher is better),",
        "computed at each volume's original resolution on validation fold "
        f"{p.get('valid_fold')} of {p.get('folds')}:",
        f"{p.get('valid_volumes')} MRI sessions from patients never seen in training "
        "(GroupKFold by patient).",
        "",
        "| Model | Score | Dice | Hausdorff (normalised, lower is better) | MLflow run |",
        "|---|---:|---:|---:|---|",
        f"| 2.5D U-Net, {p.get('encoder')} encoder | **{m['valid_score']:.4f}** | {m['valid_dice']:.4f} | "
        f"{m['valid_hausdorff']:.4f} | `{ids['model']}` |",
        f"| Predict nothing (baseline) | {b['baseline_score']:.4f} | {b['baseline_dice']:.4f} | "
        f"{b['baseline_hausdorff']:.4f} | `{ids['baseline']}` |",
        "",
        "Predicting nothing already scores well on Dice, because most slices contain no organ and an empty",
        "prediction on an empty slice counts as perfect. The Hausdorff term is what separates real models.",
        "",
        "## By organ",
        "",
        "| Organ | Score | Dice | Hausdorff |",
        "|---|---:|---:|---:|",
    ]
    for organ in cfg.classes:
        L.append(
            f"| {organ.replace('_', ' ')} | {m[f'valid_{organ}_score']:.4f} "
            f"| {m[f'valid_{organ}_dice']:.4f} | "
            f"{m[f'valid_{organ}_hausdorff']:.4f} |"
        )
    hist = meta["history"]
    best = max(hist, key=lambda h: h["val_dice_256"])
    L += [
        "",
        "## Training (from the run's parameters and history)",
        "",
        f"- {p.get('arch')} with a {p.get('encoder')} encoder ({p.get('encoder_weights')} weights), "
        f"input {p.get('image_size')} px, 2.5D context stride {p.get('context_stride')}",
        f"- {p.get('epochs')} epochs, batch {p.get('batch_size')}, peak learning rate {p.get('lr')}, "
        f"mixed precision {p.get('amp')}, on {model.data.tags.get('gpu')}",
        f"- Best epoch {best['epoch']} (validation Dice at 256 px {best['val_dice_256']:.4f}); "
        f"total time {meta['minutes']:.0f} minutes",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    (PROJECT_ROOT / "RESULTS.md").write_text(build(cfg), encoding="utf-8")
    print(f"wrote {PROJECT_ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
