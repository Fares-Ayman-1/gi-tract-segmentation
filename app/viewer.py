"""Streamlit viewer: scan, ground truth and model prediction side by side for held-out patients.

    uv run streamlit run app/viewer.py
Research demo only: not a medical device.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gisseg.metrics import dice  # noqa: E402
from gisseg.volume import stack_25d, unpack  # noqa: E402

CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "data" / "cache"))
MODELS = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
REPORTS = Path(os.environ.get("REPORTS_DIR", ROOT / "reports"))
COLOURS = {"large_bowel": (201, 154, 63), "small_bowel": (79, 120, 224), "stomach": (47, 163, 122)}

st.set_page_config(page_title="GI tract segmentation", layout="wide")


@st.cache_resource
def session():  # type: ignore[no-untyped-def]
    import onnxruntime as ort

    meta = json.loads((MODELS / "model.json").read_text(encoding="utf-8"))
    return ort.InferenceSession(str(MODELS / "model.onnx"), providers=["CPUExecutionProvider"]), meta


@st.cache_data
def predict(case_day: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sess, meta = session()
    vol = np.load(CACHE / f"{case_day}_img.npy")
    true = unpack(np.load(CACHE / f"{case_day}_mask.npy"), len(meta["classes"]))
    x = (
        np.stack([stack_25d(vol, z, meta["context_stride"]) for z in range(len(vol))]).astype(np.float32)
        / 255
    )
    logits = np.concatenate([sess.run(None, {"image": x[i : i + 16]})[0] for i in range(0, len(x), 16)])
    pred = (1 / (1 + np.exp(-logits)) > meta["threshold"]).astype(np.uint8)
    return vol, true, pred


def overlay(img: np.ndarray, masks: np.ndarray, classes: list[str]) -> Image.Image:
    rgb = np.repeat(img[..., None], 3, axis=2).astype(np.float32)
    for c, name in enumerate(classes):
        m = masks[c].astype(bool)
        rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(COLOURS[name])
    return Image.fromarray(rgb.astype(np.uint8)).resize((384, 384), Image.Resampling.NEAREST)


if not (MODELS / "model.onnx").exists() or not (REPORTS / "metrics.json").exists():
    st.error("No trained model found. Run `python -m gisseg.train` first.")
    st.stop()

meta = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
_, model_meta = session()
classes = model_meta["classes"]
ev = meta["evaluation"]
st.title("GI tract segmentation on MRI")
st.caption("2.5D U-Net · held-out patients only · research demo, not a medical device")
c1, c2, c3 = st.columns(3)
c1.metric(
    "Competition score",
    f"{ev['model']['score']:.4f}",
    f"{ev['model']['score'] - ev['empty_baseline']['score']:+.4f} vs predicting nothing",
)
c2.metric("Dice", f"{ev['model']['dice']:.4f}")
c3.metric("Hausdorff (normalised, lower is better)", f"{ev['model']['hausdorff']:.4f}")

case_day = st.selectbox("Validation MRI session (patient never seen in training)", meta["valid_case_days"])
vol, true, pred = predict(case_day)
with_organs = [z for z in range(len(vol)) if true[z].any()] or [0]
z = st.slider("Slice", 0, len(vol) - 1, with_organs[len(with_organs) // 2])
a, b, c = st.columns(3)
a.image(Image.fromarray(vol[z]).resize((384, 384)), caption=f"Scan, slice {z}")
b.image(overlay(vol[z], true[z], classes), caption="Ground truth")
c.image(overlay(vol[z], pred[z], classes), caption="Model prediction")
st.markdown(
    " · ".join(
        f"<span style='color:rgb{COLOURS[n]}'>■</span> {n.replace('_', ' ')}: "
        f"Dice {dice(pred[z, i], true[z, i]):.2f}"
        for i, n in enumerate(classes)
    ),
    unsafe_allow_html=True,
)
st.caption(
    "Per-slice Dice at 256 px for this slice. "
    "The headline numbers are the full metric at original resolution."
)
