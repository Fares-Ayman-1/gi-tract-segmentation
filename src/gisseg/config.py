"""Typed configuration loaded from YAML, plus the shared MLflow setup."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    raw: Path
    cache: Path
    models: Path
    reports: Path

    def resolve(self, root: Path) -> Paths:
        return Paths(
            **{k: (v if v.is_absolute() else (root / v).resolve()) for k, v in self.model_dump().items()}
        )


class ModelConfig(BaseModel):
    arch: str = "Unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"


class TrainConfig(BaseModel):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    num_workers: int = Field(ge=0)
    amp: bool = True
    seed: int = 42


class Config(BaseModel):
    paths: Paths
    tracking_uri: str
    artifact_root: str
    experiment: str
    classes: list[str]
    image_size: int = Field(gt=0)
    context_stride: int = Field(ge=1)
    folds: int = Field(ge=2)
    valid_fold: int = Field(ge=0)
    model: ModelConfig
    train: TrainConfig
    threshold: float = Field(gt=0, lt=1)


def load_config(path: str | Path | None = None, root: Path = PROJECT_ROOT) -> Config:
    p = Path(path) if path else root / "configs" / "default.yaml"
    cfg = Config.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    cfg.paths = cfg.paths.resolve(root)
    return cfg


def _abs(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def setup_mlflow(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + _abs(cfg.tracking_uri, root).as_posix()
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(cfg.experiment)
    if exp is not None:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        return str(exp.experiment_id)
    art = _abs(cfg.artifact_root, root)
    art.mkdir(parents=True, exist_ok=True)
    exp_id = mlflow.create_experiment(cfg.experiment, artifact_location=art.as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return str(exp_id)
