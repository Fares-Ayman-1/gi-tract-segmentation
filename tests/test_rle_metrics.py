from __future__ import annotations

import numpy as np
import pytest

from gisseg.metrics import combine, dice, hausdorff_3d, volume_scores
from gisseg.rle import decode, encode


def test_rle_hand_example() -> None:
    m = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)  # row-major: 0 1 1 0 0 1 -> starts 2 and 6
    assert encode(m) == "2 2 6 1"
    assert (decode("2 2 6 1", (2, 3)) == m).all()


def test_rle_roundtrip_and_empty() -> None:
    rng = np.random.default_rng(0)
    m = (rng.random((40, 57)) > 0.7).astype(np.uint8)
    assert (decode(encode(m), m.shape) == m).all()
    assert encode(np.zeros((3, 3))) == ""
    assert not decode(float("nan"), (3, 3)).any()


def test_dice_hand() -> None:
    a = np.zeros((4, 4))
    b = np.zeros((4, 4))
    a[0, :2] = 1  # 2 pixels
    b[0, 1:4] = 1  # 3 pixels, overlap 1
    assert dice(a, b) == pytest.approx(2 * 1 / 5)
    assert dice(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0  # both empty


def test_hausdorff_hand() -> None:
    p = np.zeros((1, 10, 10), dtype=np.uint8)
    t = np.zeros((1, 10, 10), dtype=np.uint8)
    p[0, 0, 0] = 1
    t[0, 3, 4] = 1  # distance 5; diagonal of a 1x10x10 volume = sqrt(201)
    assert hausdorff_3d(p, t) == pytest.approx(5 / np.sqrt(201))
    assert hausdorff_3d(np.zeros_like(p), np.zeros_like(t)) == 0.0
    assert hausdorff_3d(p, np.zeros_like(t)) == 1.0


def test_perfect_prediction_scores_one() -> None:
    rng = np.random.default_rng(1)
    true = (rng.random((5, 3, 16, 16)) > 0.8).astype(np.uint8)
    s = combine([volume_scores(true.copy(), true)])
    assert s["dice"] == pytest.approx(1.0)
    assert s["hausdorff"] == pytest.approx(0.0)
    assert s["score"] == pytest.approx(1.0)


def test_combine_weights_by_counts() -> None:
    a = {"dice": 1.0, "hausdorff": 0.0, "n_dice": 30, "n_hd": 3}
    b = {"dice": 0.0, "hausdorff": 1.0, "n_dice": 10, "n_hd": 3}
    s = combine([a, b])
    assert s["dice"] == pytest.approx(0.75)
    assert s["hausdorff"] == pytest.approx(0.5)
    assert s["score"] == pytest.approx(0.4 * 0.75 + 0.6 * 0.5)
