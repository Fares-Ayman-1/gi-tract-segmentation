"""End to end on synthetic data: train 1 epoch on CPU -> evaluate -> ONNX -> API contract."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gisseg.config import Config
from gisseg.rle import decode


def test_training_outputs(trained: tuple[Path, Config, dict[str, Any]]) -> None:
    _, cfg, meta = trained
    ev = meta["evaluation"]
    for k in ("model", "empty_baseline"):
        assert 0.0 <= ev[k]["score"] <= 1.0
    assert set(ev["per_class"]) == set(cfg.classes)
    assert (cfg.paths.models / "model.onnx").exists() and (cfg.paths.models / "model.json").exists()
    assert set(meta["mlflow_run_ids"]) == {"baseline", "model"}


@pytest.fixture(scope="module")
def client(trained: tuple[Path, Config, dict[str, Any]]) -> Iterator[TestClient]:
    _, cfg, _ = trained
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(cfg.paths.models))
    mp.setenv("REPORTS_DIR", str(cfg.paths.reports))
    from gisseg.serve import api

    api.get_model.cache_clear()
    with TestClient(api.app) as c:
        yield c
    api.get_model.cache_clear()
    mp.undo()


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def slices(project_raw: Path, n: int) -> list[tuple[str, tuple[str, bytes, str]]]:
    files = sorted((project_raw / "train").rglob("*.png"))[3 : 3 + n]
    return [("slices", (f.name, f.read_bytes(), "image/png")) for f in files]


def test_segment_three_slices(client: TestClient, trained: tuple[Path, Config, dict[str, Any]]) -> None:
    _, cfg, _ = trained
    r = client.post("/v1/segmentations", files=slices(cfg.paths.raw, 3))
    assert r.status_code == 200
    body = r.json()
    assert (body["height"], body["width"], body["slices_used"]) == (48, 56, 3)
    assert [o["name"] for o in body["organs"]] == cfg.classes
    for o in body["organs"]:
        m = decode(o["rle"], (48, 56))
        assert int(m.sum()) == o["area_px"] and o["present"] == (o["area_px"] > 0)
    assert "not a medical device" in body["disclaimer"].lower()


def test_segment_one_slice_and_model_info(
    client: TestClient, trained: tuple[Path, Config, dict[str, Any]]
) -> None:
    _, cfg, _ = trained
    assert client.post("/v1/segmentations", files=slices(cfg.paths.raw, 1)).json()["slices_used"] == 1
    info = client.get("/v1/model").json()
    assert info["classes"] == cfg.classes and info["validation"]["model"] is not None


def assert_error(resp, status: int, code: str) -> None:  # type: ignore[no-untyped-def]
    assert resp.status_code == status and resp.json()["error"]["code"] == code


def test_errors(client: TestClient, trained: tuple[Path, Config, dict[str, Any]]) -> None:
    _, cfg, _ = trained
    assert_error(client.post("/v1/segmentations", files=slices(cfg.paths.raw, 2)), 422, "invalid_slice_count")
    assert_error(
        client.post("/v1/segmentations", files=[("slices", ("x.png", b"not an image", "image/png"))]),
        422,
        "invalid_image",
    )
    mixed = [
        ("slices", ("a.png", png(np.zeros((10, 10), np.uint16)), "image/png")),
        ("slices", ("b.png", png(np.zeros((12, 10), np.uint16)), "image/png")),
        ("slices", ("c.png", png(np.zeros((10, 10), np.uint16)), "image/png")),
    ]
    assert_error(client.post("/v1/segmentations", files=mixed), 422, "size_mismatch")
    assert_error(client.post("/v1/segmentations"), 422, "validation_error")
    assert_error(client.get("/v1/segmentations"), 405, "method_not_allowed")


def test_volume_normalisation(client: TestClient, trained: tuple[Path, Config, dict[str, Any]]) -> None:
    _, cfg, _ = trained
    files = slices(cfg.paths.raw, 3)
    assert client.post("/v1/segmentations", files=files).json()["normalisation"] == "slices"
    r = client.post("/v1/segmentations", files=files, data={"volume_p995": "1500"})
    assert r.status_code == 200 and r.json()["normalisation"] == "volume"
    assert_error(
        client.post("/v1/segmentations", files=files, data={"volume_p995": "-1"}), 422, "validation_error"
    )
