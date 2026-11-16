"""NumPy-only volume helpers shared by training, serving and the viewer (no PyTorch, no OpenCV)."""

from __future__ import annotations

import numpy as np


def pack(masks: np.ndarray) -> np.ndarray:
    """(..., C, H, W) binary -> (..., H, W) uint8 bit flags (bit c = class c)."""
    out = np.zeros(masks.shape[:-3] + masks.shape[-2:], dtype=np.uint8)
    for c in range(masks.shape[-3]):
        out |= masks[..., c, :, :].astype(np.uint8) << c
    return out


def unpack(bits: np.ndarray, n_classes: int) -> np.ndarray:
    """(..., H, W) bit flags -> (..., C, H, W) uint8 0/1."""
    return np.stack([(bits >> c) & 1 for c in range(n_classes)], axis=-3).astype(np.uint8)


def stack_25d(volume: np.ndarray, z: int, stride: int) -> np.ndarray:
    """Slices (z - stride, z, z + stride) as channels, clamped at the volume edges -> (3, H, W)."""
    last = volume.shape[0] - 1
    idx = [min(max(z + o, 0), last) for o in (-stride, 0, stride)]
    return np.stack([volume[i] for i in idx])
