from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gisseg.config import Config
from gisseg.data.dataset import SliceDataset, folds, stack_25d
from gisseg.data.prepare import pack, unpack
from gisseg.rle import decode


def test_pack_unpack_roundtrip() -> None:
    m = (np.random.default_rng(0).random((4, 3, 8, 9)) > 0.5).astype(np.uint8)
    assert (unpack(pack(m), 3) == m).all()


def test_stack_25d_clamps_at_edges() -> None:
    vol = np.arange(5)[:, None, None] * np.ones((5, 2, 2))
    assert stack_25d(vol, 0, 2)[:, 0, 0].tolist() == [0, 0, 2]
    assert stack_25d(vol, 4, 2)[:, 0, 0].tolist() == [2, 4, 4]


def test_cache_matches_raw_masks(project: tuple[Path, Config]) -> None:
    _, cfg = project
    index = pd.read_csv(cfg.paths.cache / "index.csv")
    assert len(index) == 6 and set(index["height"]) == {48} and set(index["width"]) == {56}
    row = index.iloc[0]
    img = np.load(cfg.paths.cache / f"{row.case_day}_img.npy")
    full = unpack(np.load(cfg.paths.cache / f"{row.case_day}_mask_full.npy"), 3)
    assert img.shape == (10, 32, 32) and img.dtype == np.uint8
    assert full.shape == (10, 3, 48, 56)
    labels = pd.read_csv(cfg.paths.raw / "train.csv")
    rle = labels[(labels["id"] == f"{row.case_day}_slice_0005") & (labels["class"] == "stomach")][
        "segmentation"
    ].iloc[0]
    assert (full[4, 2] == decode(rle, (48, 56))).all()  # slice 5 (index 4), stomach (class 2)


def test_folds_never_split_a_patient(project: tuple[Path, Config]) -> None:
    _, cfg = project
    index = pd.read_csv(cfg.paths.cache / "index.csv")
    index["fold"] = folds(index, cfg.folds)
    assert (index.groupby("case")["fold"].nunique() == 1).all()
    assert set(index["fold"]) == set(range(cfg.folds))


def test_dataset_item(project: tuple[Path, Config]) -> None:
    _, cfg = project
    index = pd.read_csv(cfg.paths.cache / "index.csv")
    ds = SliceDataset(cfg.paths.cache, index["case_day"].tolist()[:2], cfg.context_stride, 3)
    assert len(ds) == 20
    item = ds[4]
    assert tuple(item["image"].shape) == (3, 32, 32) and float(item["image"].max()) <= 1.0
    assert tuple(item["mask"].shape) == (3, 32, 32)
