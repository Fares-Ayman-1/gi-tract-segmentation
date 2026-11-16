"""2.5D slice dataset over the memory-mapped cache, plus patient-level folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from gisseg.volume import stack_25d, unpack


def folds(index: pd.DataFrame, n_folds: int) -> pd.Series:
    """Fold number per volume, grouped by patient (case): no patient in both train and validation."""
    fold = pd.Series(-1, index=index.index)
    for f, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(index, groups=index["case"])):
        fold.iloc[va] = f
    return fold


class SliceDataset(Dataset[dict[str, Any]]):
    def __init__(self, cache: Path, case_days: list[str], stride: int, n_classes: int, aug: Any = None):
        self.cache, self.stride, self.n_classes, self.aug = Path(cache), stride, n_classes, aug
        self.items: list[tuple[str, int]] = []
        for cd in case_days:
            n = np.load(self.cache / f"{cd}_img.npy", mmap_mode="r").shape[0]
            self.items += [(cd, z) for z in range(n)]
        self._open: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _arrays(self, cd: str) -> tuple[np.ndarray, np.ndarray]:
        if cd not in self._open:  # opened lazily inside each worker process
            self._open[cd] = (
                np.load(self.cache / f"{cd}_img.npy", mmap_mode="r"),
                np.load(self.cache / f"{cd}_mask.npy", mmap_mode="r"),
            )
        return self._open[cd]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        cd, z = self.items[i]
        img_vol, mask_vol = self._arrays(cd)
        img = stack_25d(img_vol, z, self.stride).transpose(1, 2, 0)  # H, W, 3
        mask = unpack(np.asarray(mask_vol[z]), self.n_classes).transpose(1, 2, 0)  # H, W, C
        if self.aug is not None:
            r = self.aug(image=img, mask=mask)
            img, mask = r["image"], r["mask"]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0,
            "mask": torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float(),
            "case_day": cd,
            "z": z,
        }


def train_augs() -> Any:
    import albumentations as A

    # no flips: left/right anatomy matters (the stomach sits on one side)
    return A.Compose(
        [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-12, 12), p=0.7),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        ]
    )
