from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import write_config, write_raw

from gisseg.config import Config, load_config


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Config]:
    root = tmp_path_factory.mktemp("gi")
    write_raw(root / "data" / "raw")
    cfg = load_config(write_config(root), root=root)
    from gisseg.data.prepare import prepare

    prepare(cfg)
    return root, cfg


@pytest.fixture(scope="session")
def trained(project: tuple[Path, Config]) -> tuple[Path, Config, dict[str, Any]]:
    from gisseg.train import run

    root, cfg = project
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    meta = run(cfg, epochs=1, root=root, weights=None)
    return root, cfg, meta
