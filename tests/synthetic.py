"""Tiny synthetic dataset in the competition's exact formats.

16-bit PNG slices named slice_XXXX_<width>_<height>_<sx>_<sy>.png, and train.csv with RLE masks. Images
are deliberately NOT square (48 high x 56 wide), so a height/width mix-up fails the tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from gisseg.rle import encode

CLASSES = ["large_bowel", "small_bowel", "stomach"]
H, W, SLICES = 48, 56, 10


def write_raw(raw: Path, n_cases: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    yy, xx = np.mgrid[:H, :W]
    for case in range(1, n_cases + 1):
        cd = f"case{case}_day{case % 3}"
        scans = raw / "train" / f"case{case}" / cd / "scans"
        scans.mkdir(parents=True, exist_ok=True)
        for s in range(1, SLICES + 1):
            img = rng.normal(600, 80, (H, W))
            masks = np.zeros((3, H, W), dtype=np.uint8)
            if 3 <= s <= 7:  # stomach: a disc that grows and shrinks across slices
                r = 4 + (2 - abs(s - 5))
                masks[2] = (yy - 14) ** 2 + (xx - 38) ** 2 <= r**2
            if s >= 5:  # large bowel: a band in the lower part
                masks[0, 30:38, 8:30] = 1
            img = img + 900.0 * masks[2] + 500.0 * masks[0]
            Image.fromarray(np.clip(img, 0, 65535).astype(np.uint16)).save(
                scans / f"slice_{s:04d}_{W}_{H}_1.50_1.50.png"
            )
            for c, name in enumerate(CLASSES):
                rows.append(
                    {
                        "id": f"{cd}_slice_{s:04d}",
                        "class": name,
                        "segmentation": encode(masks[c]) if masks[c].any() else np.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(raw / "train.csv", index=False)


def write_config(root: Path) -> Path:
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text("utf-8")
    )
    cfg.update(
        {
            "tracking_uri": str((root / "mlflow.db").resolve()),
            "artifact_root": str((root / "mlruns").resolve()),
            "experiment": "test",
            "image_size": 32,
            "folds": 3,
        }
    )
    cfg["model"].update({"encoder": "resnet18", "encoder_weights": None})
    cfg["train"].update({"epochs": 1, "batch_size": 4, "num_workers": 0, "amp": False})
    p = root / "configs" / "test.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p
