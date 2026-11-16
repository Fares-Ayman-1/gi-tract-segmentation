from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gisseg.config import Config
from gisseg.data.dataset import folds
from gisseg.monitoring import check, psi, to_markdown
from gisseg.report import build


def test_psi() -> None:
    a = np.random.default_rng(0).normal(size=3000)
    assert psi(a, a.copy()) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, a + 2) > 0.2


def test_monitoring(project: tuple[Path, Config]) -> None:
    _, cfg = project
    index = pd.read_csv(cfg.paths.cache / "index.csv")
    index["fold"] = folds(index, cfg.folds)
    r = check(cfg, index[index["fold"] != 0], index[index["fold"] == 0])
    assert r["values"]["unseen_resolution_share"] == 0.0  # synthetic scans all share one size
    assert "Retrain recommended" in to_markdown(r)


def test_results_from_mlflow(trained: tuple[Path, Config, dict[str, Any]]) -> None:
    root, cfg, meta = trained
    md = build(cfg, root)
    assert f"`{meta['mlflow_run_ids']['model']}`" in md
    assert f"{meta['evaluation']['model']['score']:.4f}" in md
