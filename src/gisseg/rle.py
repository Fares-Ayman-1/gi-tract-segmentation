"""Run-length encoding used by the competition: 1-indexed (start, length) pairs over the row-major
(C-order) flattened mask."""

from __future__ import annotations

import numpy as np


def decode(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not isinstance(rle, str) or not rle.strip():
        return mask.reshape(shape)
    nums = np.asarray(rle.split(), dtype=np.int64)
    for start, length in zip(nums[0::2] - 1, nums[1::2], strict=True):
        mask[start : start + length] = 1
    return mask.reshape(shape)


def encode(mask: np.ndarray) -> str:
    flat = np.concatenate([[0], mask.reshape(-1).astype(np.uint8), [0]])
    runs = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))
