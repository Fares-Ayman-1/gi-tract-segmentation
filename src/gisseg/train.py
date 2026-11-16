"""Train the 2.5D U-Net on one patient-grouped fold, evaluate with the competition metric, export ONNX.

- Mixed precision (fp16 autocast + GradScaler) and batch 16 at 256 px fit in a 6 GB laptop GPU.
- Loss: BCE + soft Dice (multilabel: the three organs are separate channels).
- Model selection: best validation Dice at 256 px (fast, every epoch).
- Final evaluation: the competition metric at each volume's ORIGINAL resolution (predictions upsampled),
  compared with the "predict nothing" baseline (non-trivial here: most slices contain no organ).

    uv run python -m gisseg.train                  # full run
    uv run python -m gisseg.train --epochs 1 --limit 4   # quick smoke run on 4 volumes
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gisseg.config import PROJECT_ROOT, Config, load_config, setup_mlflow
from gisseg.data.dataset import SliceDataset, folds, train_augs
from gisseg.metrics import combine, volume_scores
from gisseg.volume import stack_25d, unpack


def build_model(cfg: Config, weights: str | None = "config") -> torch.nn.Module:
    w = cfg.model.encoder_weights if weights == "config" else weights
    return smp.create_model(
        cfg.model.arch,
        encoder_name=cfg.model.encoder,
        encoder_weights=w,
        in_channels=3,
        classes=len(cfg.classes),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - albumentations draws from NumPy's global generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module, cache: Path, case_day: str, stride: int, device: str, batch: int = 32
) -> np.ndarray:
    """Sigmoid probabilities for every slice of a volume at cache resolution -> (S, C, size, size) float16."""
    vol = np.load(cache / f"{case_day}_img.npy", mmap_mode="r")
    out = []
    for s in range(0, vol.shape[0], batch):
        x = np.stack([stack_25d(vol, z, stride) for z in range(s, min(s + batch, vol.shape[0]))])
        xt = torch.from_numpy(x).float().div_(255).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(xt)).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def evaluate(model: torch.nn.Module, cfg: Config, index: pd.DataFrame, device: str) -> dict[str, Any]:
    """Competition metric at original resolution, for the model and for the empty-prediction baseline."""
    model.eval()
    n_cls = len(cfg.classes)
    res_model: list[dict[str, float]] = []
    res_empty: list[dict[str, float]] = []
    per_class: dict[str, list[dict[str, float]]] = {c: [] for c in cfg.classes}
    for row in index.itertuples():
        probs = predict_volume(model, cfg.paths.cache, row.case_day, cfg.context_stride, device)
        up = F.interpolate(
            torch.from_numpy(probs.astype(np.float32)),
            size=(row.height, row.width),
            mode="bilinear",
            align_corners=False,
        ).numpy()
        pred = (up > cfg.threshold).astype(np.uint8)
        true = unpack(np.load(cfg.paths.cache / f"{row.case_day}_mask_full.npy"), n_cls)
        res_model.append(volume_scores(pred, true))
        res_empty.append(volume_scores(np.zeros_like(true), true))
        for c, name in enumerate(cfg.classes):
            per_class[name].append(volume_scores(pred[:, c : c + 1], true[:, c : c + 1]))
    return {
        "model": combine(res_model),
        "empty_baseline": combine(res_empty),
        "per_class": {k: combine(v) for k, v in per_class.items()},
    }


def export_onnx(model: torch.nn.Module, cfg: Config, path: Path) -> None:
    model = model.float().cpu().eval()
    dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    (path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "classes": cfg.classes,
                "image_size": cfg.image_size,
                "context_stride": cfg.context_stride,
                "threshold": cfg.threshold,
                "normalisation": "per-volume 99.5th percentile, scaled to 0-1",
                "input": "float32 (N, 3, size, size): slices z-stride, z, z+stride",
                "output": "logits (N, C, size, size)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run(
    cfg: Config,
    epochs: int | None = None,
    limit: int | None = None,
    root: Path = PROJECT_ROOT,
    weights: str | None = "config",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(cfg.paths.cache / "index.csv")
    index["fold"] = folds(index, cfg.folds)
    if limit:
        index = index.groupby("fold", group_keys=False).head(max(1, limit // cfg.folds))
    tr_idx, va_idx = index[index["fold"] != cfg.valid_fold], index[index["fold"] == cfg.valid_fold]
    n_cls = len(cfg.classes)
    train_ds = SliceDataset(
        cfg.paths.cache, tr_idx["case_day"].tolist(), cfg.context_stride, n_cls, train_augs()
    )
    valid_ds = SliceDataset(cfg.paths.cache, va_idx["case_day"].tolist(), cfg.context_stride, n_cls)
    kw: dict[str, Any] = dict(
        num_workers=cfg.train.num_workers,
        pin_memory=False,  # pinned buffers count against the Windows commit limit
        persistent_workers=cfg.train.num_workers > 0,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **kw)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.train.batch_size * 2, shuffle=False, **kw)

    model = build_model(cfg, weights).to(device)
    n_epochs = epochs or cfg.train.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total = max(n_epochs * len(train_dl), 4)
    # warm up for 10% of steps, but always at least 2 (very short runs would otherwise divide by zero)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=total, pct_start=max(0.1, 2 / total)
    )
    amp = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bce, soft_dice = torch.nn.BCEWithLogitsLoss(), smp.losses.DiceLoss(mode="multilabel", from_logits=True)
    print(
        f"device {device} | train {len(train_ds):,} slices ({len(tr_idx)} volumes) | "
        f"valid {len(valid_ds):,} slices ({len(va_idx)} volumes) | {n_epochs} epochs",
        flush=True,
    )

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for ep in range(1, n_epochs + 1):
        model.train()
        te, losses = time.perf_counter(), []
        for b in train_dl:
            x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = bce(logits, y) + soft_dice(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        model.eval()
        inter = union = 0.0
        dices: list[float] = []
        with torch.no_grad():
            for b in valid_dl:
                x, y = b["image"].to(device), b["mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    p = (torch.sigmoid(model(x)) > cfg.threshold).float()
                inter_b = (p * y).sum(dim=(2, 3))
                denom = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
                d = torch.where(denom == 0, torch.ones_like(denom), 2 * inter_b / denom.clamp(min=1))
                dices += d.flatten().tolist()
                inter += float(inter_b.sum())
                union += float(denom.sum())
        val_dice = float(np.mean(dices))
        history.append(
            {
                "epoch": ep,
                "train_loss": float(np.mean(losses)),
                "val_dice_256": val_dice,
                "seconds": time.perf_counter() - te,
            }
        )
        print(
            f"epoch {ep}/{n_epochs}: loss {history[-1]['train_loss']:.4f} | val Dice@256 {val_dice:.4f} | "
            f"{history[-1]['seconds']:.0f}s",
            flush=True,
        )
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), cfg.paths.models / "best.pt")

    model.load_state_dict(torch.load(cfg.paths.models / "best.pt", map_location=device))
    ev = evaluate(model, cfg, va_idx, device)
    print(json.dumps({k: ev[k] for k in ("model", "empty_baseline")}, indent=2), flush=True)
    export_onnx(model, cfg, cfg.paths.models / "model.onnx")

    setup_mlflow(cfg, root)
    tags = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none"}
    run_ids = {}
    with mlflow.start_run(run_name="empty-baseline", tags=tags) as r:
        mlflow.log_metrics({f"baseline_{k}": v for k, v in ev["empty_baseline"].items()})
        run_ids["baseline"] = r.info.run_id
    with mlflow.start_run(run_name="unet-2.5d", tags=tags) as r:
        mlflow.log_params(
            {
                "arch": cfg.model.arch,
                "encoder": cfg.model.encoder,
                "encoder_weights": str(weights if weights != "config" else cfg.model.encoder_weights),
                "image_size": cfg.image_size,
                "context_stride": cfg.context_stride,
                "epochs": n_epochs,
                "batch_size": cfg.train.batch_size,
                "lr": cfg.train.lr,
                "valid_fold": cfg.valid_fold,
                "folds": cfg.folds,
                "train_volumes": len(tr_idx),
                "valid_volumes": len(va_idx),
                "amp": amp,
                "threshold": cfg.threshold,
            }
        )
        for h in history:
            mlflow.log_metrics(
                {"train_loss": h["train_loss"], "val_dice_256": h["val_dice_256"]}, step=int(h["epoch"])
            )
        mlflow.log_metrics({f"valid_{k}": v for k, v in ev["model"].items()})
        for name, s in ev["per_class"].items():
            mlflow.log_metrics({f"valid_{name}_{k}": v for k, v in s.items()})
        run_ids["model"] = r.info.run_id

    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluation": ev,
        "history": history,
        "mlflow_run_ids": run_ids,
        "valid_case_days": va_idx["case_day"].tolist(),
        "minutes": (time.perf_counter() - t0) / 60,
    }
    (cfg.paths.reports / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done in {meta['minutes']:.1f} min; runs {run_ids}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only this many volumes (smoke test)")
    ap.add_argument(
        "--workers", type=int, default=None, help="DataLoader workers (Windows: each one loads CUDA)"
    )
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.workers is not None:
        cfg.train.num_workers = a.workers
    run(cfg, a.epochs, a.limit)


if __name__ == "__main__":
    main()
