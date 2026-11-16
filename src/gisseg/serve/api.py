"""Segmentation API (ONNX Runtime, no PyTorch at serving time).

POST one MRI slice, or three consecutive ones (z - 2, z, z + 2) for the 2.5D model, as PNG (8- or 16-bit).
The response gives, for each organ, whether it is present, its area in pixels and its mask as competition
RLE at the input resolution. Research demo only: not a medical device.

    MODEL_DIR=models uv run uvicorn gisseg.serve.api:app --port 8020
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from gisseg import __version__
from gisseg.rle import encode

ROOT = Path(__file__).resolve().parents[3]
MAX_BYTES = 5 * 1024 * 1024


class Segmenter:
    def __init__(self, model_dir: Path, reports: Path):
        import onnxruntime as ort

        onnx = model_dir / "model.onnx"
        if not onnx.exists():
            raise FileNotFoundError(f"no model in {model_dir}; run `python -m gisseg.train` first")
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.meta: dict[str, Any] = json.loads(onnx.with_suffix(".json").read_text(encoding="utf-8"))
        m = reports / "metrics.json"
        self.metrics: dict[str, Any] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}

    def predict(
        self, slices: list[np.ndarray], volume_p995: float | None = None
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """slices: 1 or 3 arrays (H, W) of raw intensities -> binary masks (C, H, W) at input resolution.

        Training divided every session by its 99.5th-percentile intensity. Passing that value as
        ``volume_p995`` reproduces the training input exactly; without it the percentile of the given
        slices is used, which can differ (e.g. stomach Dice 0.62 -> 0.29 on one held-out slice).
        """
        if len(slices) == 1:
            slices = slices * 3
        h, w = slices[1].shape
        vol = np.stack(slices).astype(np.float32)
        hi = volume_p995 or float(np.percentile(vol, 99.5)) or 1.0
        vol = np.clip(vol / hi, 0, 1)
        size = int(self.meta["image_size"])
        x = np.stack(
            [np.asarray(Image.fromarray(v).resize((size, size), Image.Resampling.BILINEAR)) for v in vol]
        )
        logits = self.session.run(None, {"image": x[None].astype(np.float32)})[0][0]
        probs = 1 / (1 + np.exp(-logits))
        up = np.stack(
            [
                np.asarray(Image.fromarray(p.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR))
                for p in probs
            ]
        )
        return (up > float(self.meta["threshold"])).astype(np.uint8), (h, w)


@lru_cache(maxsize=1)
def get_model() -> Segmenter:
    return Segmenter(
        Path(os.environ.get("MODEL_DIR", ROOT / "models")),
        Path(os.environ.get("REPORTS_DIR", ROOT / "reports")),
    )


# ------------------------------------------------------------------------------------- schemas
class ErrorBody(BaseModel):
    code: str = Field(examples=["invalid_image"])
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class OrganMask(BaseModel):
    name: str = Field(examples=["stomach"])
    present: bool
    area_px: int
    rle: str = Field(
        description="Competition RLE (1-indexed start/length pairs, row-major) at input resolution"
    )


class SegmentationResponse(BaseModel):
    height: int
    width: int
    slices_used: int = Field(description="1 = the slice is repeated as context; 3 = true 2.5D input")
    normalisation: str = Field(
        description="`volume` when volume_p995 was given (matches training), else `slices`"
    )
    organs: list[OrganMask]
    disclaimer: str = "Research demo only. Not a medical device and not for clinical use."


class ModelInfo(BaseModel):
    api_version: str
    model: str
    classes: list[str]
    image_size: int
    validation: dict[str, Any] = Field(description="Competition metric on the held-out patients")
    mlflow_run_ids: dict[str, str]


# ------------------------------------------------------------------------------------- app + errors
app = FastAPI(
    title="GI Tract Segmentation API",
    version=__version__,
    description="2.5D U-Net segmentation of the stomach, small bowel and large bowel on abdominal MRI "
    'slices. Errors always use `{"error": {"code", "message", "details"}}`. Research demo only.',
)
ERRORS: dict[int | str, dict[str, Any]] = {
    413: {"model": ErrorResponse, "description": "`file_too_large`"},
    422: {
        "model": ErrorResponse,
        "description": "`validation_error`, `invalid_slice_count`, `invalid_image`, `size_mismatch`",
    },
}


def _err(status: int, code: str, message: str, details: list[dict[str, Any]] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message, "details": details}}
    )


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"field": ".".join(str(p) for p in e["loc"]), "issue": e["msg"]} for e in exc.errors()]
    return _err(422, "validation_error", "The request is invalid.", details)


@app.exception_handler(StarletteHTTPException)
async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return _err(exc.status_code, exc.detail["code"], exc.detail["message"], exc.detail.get("details"))
    codes = {404: "not_found", 405: "method_not_allowed"}
    return _err(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
    return _err(500, "internal_error", "Something went wrong on our side.")


def _fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


@app.get("/health", tags=["meta"], summary="Liveness check")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/model", tags=["meta"], summary="Model details and validation scores", response_model=ModelInfo)
def model_info() -> ModelInfo:
    m = get_model()
    ev = m.metrics.get("evaluation", {})
    return ModelInfo(
        api_version=__version__,
        model="2.5D U-Net (ResNet-34 encoder), ONNX Runtime",
        classes=m.meta["classes"],
        image_size=m.meta["image_size"],
        validation={"model": ev.get("model"), "empty_baseline": ev.get("empty_baseline")},
        mlflow_run_ids=m.metrics.get("mlflow_run_ids", {}),
    )


@app.post(
    "/v1/segmentations",
    tags=["segmentation"],
    summary="Segment organs on 1 or 3 MRI slices",
    response_model=SegmentationResponse,
    responses=ERRORS,
)
async def segment(
    slices: Annotated[
        list[UploadFile], File(description="1 slice, or 3 consecutive slices (z-2, z, z+2), PNG")
    ],
    volume_p995: Annotated[
        float | None,
        Form(
            gt=0,
            description="99.5th-percentile intensity of the whole MRI session. Strongly recommended: "
            "it reproduces the training normalisation. Without it the given slices are used, "
            "which is less accurate.",
        ),
    ] = None,
) -> SegmentationResponse:
    if len(slices) not in (1, 3):
        raise _fail(422, "invalid_slice_count", f"Send 1 or 3 slices, got {len(slices)}.")
    arrays = []
    for f in slices:
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise _fail(413, "file_too_large", f"{f.filename} is larger than {MAX_BYTES // 1024 // 1024} MB.")
        try:
            with Image.open(io.BytesIO(data)) as im:
                arr = np.asarray(im.convert("I") if im.mode not in ("I;16", "I", "L") else im).astype(
                    np.float32
                )
        except Exception as e:
            raise _fail(422, "invalid_image", f"{f.filename} is not a readable image.") from e
        if arr.ndim != 2:
            raise _fail(422, "invalid_image", f"{f.filename} must be a single-channel (grayscale) slice.")
        arrays.append(arr)
    if len({a.shape for a in arrays}) > 1:
        raise _fail(422, "size_mismatch", "All slices must have the same height and width.")
    masks, (h, w) = get_model().predict(arrays, volume_p995)
    organs = [
        OrganMask(name=n, present=bool(m.any()), area_px=int(m.sum()), rle=encode(m))
        for n, m in zip(get_model().meta["classes"], masks, strict=True)
    ]
    return SegmentationResponse(
        height=h,
        width=w,
        slices_used=len(arrays),
        organs=organs,
        normalisation="volume" if volume_p995 else "slices",
    )
