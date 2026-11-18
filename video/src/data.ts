// Every number on screen comes from the project's own files: reports/*.json, models/model.json, or a
// README line cited next to the constant. public/data/slices.json is written by video/scripts/extract.py
// (real slices, real ground truth, real ONNX predictions).
import metricsJson from "../../reports/metrics.json";
import monitoringJson from "../../reports/monitoring.json";
import modelJson from "../../models/model.json";
import slicesJson from "../public/data/slices.json";

export type Organ = "large_bowel" | "small_bowel" | "stomach";
export type Poly = number[][]; // [[x, y], ...] in image pixels
export type Outlines = Record<Organ, Poly[]>;
type Score = { dice: number; hausdorff: number; score: number };

const m = metricsJson as unknown as {
  evaluation: { model: Score; empty_baseline: Score; per_class: Record<Organ, Score> };
  history: { epoch: number; train_loss: number; val_dice_256: number; seconds: number }[];
  valid_case_days: string[];
  minutes: number;
  mlflow_run_ids: Record<string, string>;
};
const mon = monitoringJson as {
  reference_volumes: number;
  current_volumes: number;
  values: { intensity_psi: number; unseen_resolution_share: number; prevalence_change: number };
  alerts: Record<string, boolean>;
  retrain_recommended: boolean;
};
const meta = modelJson as { classes: Organ[]; image_size: number; context_stride: number; threshold: number };

export type SliceBase = { case_day: string; z: number; size: number; png: string; truth: Outlines };
const s = slicesJson as unknown as {
  classes: Organ[];
  hook: SliceBase & { context: string[]; context_z: number[]; pred: Outlines; dice: Record<Organ, number> };
  insight: SliceBase & { pred_volume: Outlines; pred_slices: Outlines; dice_volume: Record<Organ, number>; dice_slices: Record<Organ, number> };
  miss: SliceBase & { pred: Outlines; dice: Record<Organ, number> };
};
export const SLICES = s;

export const ORGANS: Organ[] = meta.classes;
export const ORGAN_NAME: Record<Organ, string> = { large_bowel: "Large bowel", small_bowel: "Small bowel", stomach: "Stomach" };

const ev = m.evaluation;
const best = m.history.reduce((a, h) => (h.val_dice_256 > a.val_dice_256 ? h : a));
const dip = m.history.reduce((a, h, i) => (i > 0 && h.val_dice_256 - m.history[i - 1].val_dice_256 < a.drop ? { drop: h.val_dice_256 - m.history[i - 1].val_dice_256, h } : a), {
  drop: 0,
  h: m.history[0],
}).h;

export const NUM = {
  // README line 5 (verified against data/cache/index.csv: 274 rows, 85 cases, 38,496 slices)
  sessions: 274,
  patients: 85,
  slices: 38496,
  // reports/metrics.json + reports/monitoring.json
  validSessions: m.valid_case_days.length,
  validPatients: new Set(m.valid_case_days.map((c) => c.split("_")[0])).size,
  trainSessions: mon.reference_volumes,
  score: ev.model.score,
  dice: ev.model.dice,
  hausdorff: ev.model.hausdorff,
  baseScore: ev.empty_baseline.score,
  baseDice: ev.empty_baseline.dice,
  baseHausdorff: ev.empty_baseline.hausdorff,
  perClass: ev.per_class,
  history: m.history,
  epochs: m.history.length,
  minutes: m.minutes,
  best,
  dip,
  runIds: m.mlflow_run_ids,
  // models/model.json
  size: meta.image_size,
  stride: meta.context_stride,
  threshold: meta.threshold,
  // reports/monitoring.json
  psi: mon.values.intensity_psi,
  unseen: mon.values.unseen_resolution_share,
  prevalence: mon.values.prevalence_change,
  retrain: mon.retrain_recommended,
  // README, "Why volume_p995 matters" (line 121-122): case108_day0 slice 73, stomach Dice
  p995Without: 0.245,
  p995With: 0.604,
  // README, "The miss" (line 61-62): case108_day0 slice 78, stomach Dice
  missDice: 0.0,
  // README line 49: training hardware
  gpu: "RTX 3050 Laptop GPU (6 GB)",
};

export const f2 = (v: number) => v.toFixed(2);
export const f3 = (v: number) => v.toFixed(3);
export const f4 = (v: number) => v.toFixed(4);
