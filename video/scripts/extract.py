"""Export the handful of MRI slices the video shows, with real ground truth and real ONNX predictions.

Run from the project root (CPU only, about a minute):

    .venv/Scripts/python video/scripts/extract.py

Writes video/public/data/*.png (grayscale slices) and video/public/data/slices.json (organ outlines as
polygons in pixel coordinates, per-slice Dice). public/data/ is git-ignored: it is derived from the
competition data (DATA_RULES.md: only a few slices, never a whole scan).

Slices on screen (5 of at most 8):
- case117_day16, z 34 / 36 / 38: the hook (z = 36) and the 2.5D input (z - 2, z, z + 2), from the 256 px
  cache, predicted exactly as the viewer does (cached slices / 255 into model.onnx).
- case108_day0, z 73: the volume_p995 insight. Raw PNGs into the API's own ``Segmenter.predict`` with and
  without the session's 99.5th percentile (README: stomach Dice 0.245 -> 0.604).
- case108_day0, z 78: the honest miss (README: stomach Dice 0.00), same API path with volume_p995.
z is the 0-based slice index, as in the viewer's slider and the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gisseg.data.prepare import normalise_volume, read_scan  # noqa: E402
from gisseg.metrics import dice  # noqa: E402
from gisseg.serve.api import Segmenter  # noqa: E402
from gisseg.volume import stack_25d, unpack  # noqa: E402

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw" / "train"
OUT = ROOT / "video" / "public" / "data"
HOOK = ("case117_day16", 36)
MISS_CASE = "case108_day0"
INSIGHT_Z, MISS_Z = 73, 78


def polygons(mask: np.ndarray, eps: float = 0.6, min_area: float = 6.0) -> list[list[list[float]]]:
    """Binary mask -> outer contours as [[x, y], ...] (pixel-centre coordinates), small specks dropped."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        pts = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        out.append([[float(x) + 0.5, float(y) + 0.5] for x, y in pts])
    return out


def outline(masks: np.ndarray, classes: list[str]) -> dict[str, list[list[list[float]]]]:
    return {n: polygons(masks[c]) for c, n in enumerate(classes)}


def scores(pred: np.ndarray, true: np.ndarray, classes: list[str]) -> dict[str, float]:
    return {n: round(dice(pred[c], true[c]), 3) for c, n in enumerate(classes)}


def save_png(img: np.ndarray, name: str) -> str:
    Image.fromarray(img.astype(np.uint8)).save(OUT / name)
    return f"data/{name}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(ROOT / "models", ROOT / "reports")
    classes: list[str] = seg.meta["classes"]
    stride = int(seg.meta["context_stride"])
    result: dict[str, object] = {"classes": classes, "stride": stride}

    # hook + 2.5D: the 256 px cache, normalised by the whole session exactly as in training and the viewer
    cd, z = HOOK
    vol = np.load(CACHE / f"{cd}_img.npy")
    true = unpack(np.load(CACHE / f"{cd}_mask.npy"), len(classes))[z]
    pred, _ = seg.predict(list(stack_25d(vol, z, stride)), 255.0)  # uint8 / 255 == the viewer's input
    result["hook"] = {
        "case_day": cd,
        "z": z,
        "size": int(vol.shape[1]),
        "png": save_png(vol[z], f"{cd}_{z}.png"),
        "context": [save_png(vol[z + o], f"{cd}_{z + o}.png") for o in (-stride, 0, stride)],
        "context_z": [z - stride, z, z + stride],
        "truth": outline(true, classes),
        "pred": outline(pred, classes),
        "dice": scores(pred, true, classes),
    }

    # case108_day0: raw intensities through the API, as in the README's volume_p995 experiment
    paths = sorted((RAW / MISS_CASE.split("_")[0] / MISS_CASE / "scans").glob("*.png"))
    raw = np.stack([read_scan(str(p)) for p in paths]).astype(np.float32)
    p995 = float(np.percentile(raw, 99.5))
    shown = normalise_volume(raw)  # the same display normalisation the cache uses, at original resolution
    full = unpack(np.load(CACHE / f"{MISS_CASE}_mask_full.npy"), len(classes))

    def at(z: int, p: float | None) -> np.ndarray:
        return seg.predict(list(stack_25d(raw, z, stride)), p)[0]

    pv, ps = at(INSIGHT_Z, p995), at(INSIGHT_Z, None)
    result["insight"] = {
        "case_day": MISS_CASE,
        "z": INSIGHT_Z,
        "size": int(raw.shape[1]),
        "png": save_png(shown[INSIGHT_Z], f"{MISS_CASE}_{INSIGHT_Z}.png"),
        "truth": outline(full[INSIGHT_Z], classes),
        "pred_volume": outline(pv, classes),
        "pred_slices": outline(ps, classes),
        "dice_volume": scores(pv, full[INSIGHT_Z], classes),
        "dice_slices": scores(ps, full[INSIGHT_Z], classes),
    }
    pm = at(MISS_Z, p995)
    result["miss"] = {
        "case_day": MISS_CASE,
        "z": MISS_Z,
        "size": int(raw.shape[1]),
        "png": save_png(shown[MISS_Z], f"{MISS_CASE}_{MISS_Z}.png"),
        "truth": outline(full[MISS_Z], classes),
        "pred": outline(pm, classes),
        "dice": scores(pm, full[MISS_Z], classes),
    }
    (OUT / "slices.json").write_text(json.dumps(result), encoding="utf-8")
    for k in ("hook", "insight", "miss"):
        d = result[k]
        assert isinstance(d, dict)
        print(k, d["case_day"], d["z"], {x: v for x, v in d.items() if x.startswith("dice")})
    print("wrote", OUT / "slices.json")


if __name__ == "__main__":
    main()
