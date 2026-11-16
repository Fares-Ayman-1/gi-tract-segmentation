"""Turn the raw scans + RLE masks into a fast, memory-mapped cache.

For every case-day volume (one MRI session) it writes, in ``paths.cache``:
  <case_day>_img.npy        (S, size, size) uint8   intensity-normalised slices
  <case_day>_mask.npy       (S, size, size) uint8   class masks as bit flags (bit c = class c)
  <case_day>_mask_full.npy  (S, H, W) uint8         the same at the original resolution (for the metric)
and index.csv with one row per volume. Arrays are opened with mmap, so DataLoader worker processes on
Windows share them instead of each loading the whole dataset.

Intensities are normalised per volume: divide by the 99.5th percentile, clip to [0, 1], scale to 0-255.

    uv run python -m gisseg.data.prepare
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gisseg.config import Config, load_config
from gisseg.rle import decode
from gisseg.volume import pack, unpack  # noqa: F401 - re-exported

SLICE_RE = re.compile(r"slice_(\d+)_(\d+)_(\d+)_([\d.]+)_([\d.]+)\.png$")


def read_scan(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"could not read scan {path}")
    return np.asarray(img)


def normalise_volume(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    hi = float(np.percentile(v, 99.5)) or 1.0
    return (np.clip(v / hi, 0, 1) * 255).astype(np.uint8)


def scan_index(raw: Path) -> pd.DataFrame:
    rows = []
    for p in (raw / "train").rglob("*.png"):
        m = SLICE_RE.search(p.name)
        if not m:
            continue
        case_day = p.parent.parent.name  # caseXX_dayYY
        rows.append(
            {
                "case_day": case_day,
                "case": case_day.split("_")[0],
                "slice": int(m.group(1)),
                "width": int(m.group(2)),
                "height": int(m.group(3)),
                "path": str(p),
            }
        )
    df = pd.DataFrame(rows)
    df["id"] = df["case_day"] + "_slice_" + df["slice"].map("{:04d}".format)
    return df.sort_values(["case_day", "slice"]).reset_index(drop=True)


def prepare(cfg: Config) -> pd.DataFrame:
    raw, out, size = cfg.paths.raw, cfg.paths.cache, cfg.image_size
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(raw / "train.csv")
    seg = labels.pivot(index="id", columns="class", values="segmentation")
    scans = scan_index(raw)
    t0, rows = time.perf_counter(), []
    for case_day, g in scans.groupby("case_day", sort=True):
        h, w = int(g["height"].iloc[0]), int(g["width"].iloc[0])
        imgs = normalise_volume(np.stack([read_scan(p) for p in g["path"]]))
        full = np.zeros((len(g), len(cfg.classes), h, w), dtype=np.uint8)
        for i, sid in enumerate(g["id"]):
            if sid in seg.index:
                for c, name in enumerate(cfg.classes):
                    full[i, c] = decode(seg.loc[sid, name], (h, w))
        small_img = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in imgs])
        small_mask = np.stack(
            [[cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) for m in s] for s in full]
        )
        np.save(out / f"{case_day}_img.npy", small_img)
        np.save(out / f"{case_day}_mask.npy", pack(small_mask))
        np.save(out / f"{case_day}_mask_full.npy", pack(full))
        rows.append(
            {
                "case_day": case_day,
                "case": g["case"].iloc[0],
                "n_slices": len(g),
                "height": h,
                "width": w,
                "positive_slices": int(full.any(axis=(1, 2, 3)).sum()),
            }
        )
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    print(
        f"cached {len(index)} volumes, {int(index['n_slices'].sum())} slices, "
        f"{int(index['positive_slices'].sum())} with organs, in {time.perf_counter() - t0:.0f}s"
    )
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    prepare(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
