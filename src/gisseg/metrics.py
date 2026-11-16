"""The competition metric: 0.4 x Dice + 0.6 x (1 - normalised 3D Hausdorff distance).

- Dice is computed per (slice, class) and averaged; a slice where both prediction and truth are empty
  scores 1 (this is why "predict nothing" is a non-trivial baseline here).
- Hausdorff is computed per (case-day volume, class) in voxel units and divided by the volume's diagonal,
  so it lies in [0, 1]. Both empty -> 0; exactly one empty -> 1 (the worst case).
"""

from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, true: np.ndarray) -> float:
    p, t = pred.astype(bool), true.astype(bool)
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else float(2 * (p & t).sum() / denom)


def hausdorff_3d(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalised symmetric Hausdorff distance between two binary volumes (S, H, W)."""
    p, t = pred.astype(bool), true.astype(bool)
    if not p.any() and not t.any():
        return 0.0
    if not p.any() or not t.any():
        return 1.0
    from scipy.ndimage import distance_transform_edt  # lazy: the viewer only needs dice()

    # distance of every voxel to the nearest foreground voxel of the other set
    to_t = distance_transform_edt(~t)
    to_p = distance_transform_edt(~p)
    hd = max(float(to_t[p].max()), float(to_p[t].max()))
    return min(hd / float(np.sqrt(np.sum(np.square(p.shape)))), 1.0)


def volume_scores(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Mean Dice and mean Hausdorff over classes for one case-day.

    pred, true: (S, C, H, W) binary arrays.
    """
    slices, classes = true.shape[0], true.shape[1]
    d = [dice(pred[s, c], true[s, c]) for s in range(slices) for c in range(classes)]
    h = [hausdorff_3d(pred[:, c], true[:, c]) for c in range(classes)]
    return {"dice": float(np.mean(d)), "hausdorff": float(np.mean(h)), "n_dice": len(d), "n_hd": len(h)}


def combine(volumes: list[dict[str, float]]) -> dict[str, float]:
    """Pool per-volume results.

    Dice is averaged over all (slice, class) pairs, Hausdorff over all (volume, class) pairs.
    """
    n_d = sum(v["n_dice"] for v in volumes)
    n_h = sum(v["n_hd"] for v in volumes)
    d = sum(v["dice"] * v["n_dice"] for v in volumes) / n_d
    h = sum(v["hausdorff"] * v["n_hd"] for v in volumes) / n_h
    return {"dice": d, "hausdorff": h, "score": 0.4 * d + 0.6 * (1 - h)}
