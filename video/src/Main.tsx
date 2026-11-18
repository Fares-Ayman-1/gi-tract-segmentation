import React from "react";
import scenesJson from "../scenes.json";
import { Timeline, type SceneSpec } from "./kit/Timeline";
import { SeriesOutro } from "./kit/series";
import { Demo, Engineering } from "./scenes/App";
import { Data, Hook, TwoFive, Why } from "./scenes/Intro";
import { Curve, Metric, Model } from "./scenes/Model";
import { Insight, Miss, Results } from "./scenes/Results";

export const SCENES = scenesJson as SceneSpec[];

// Week number of this project in the series (1-4).
const WEEK = 2;

const registry: Record<string, React.FC<{ dur: number }>> = {
  hook: Hook, // series title merged into the hook
  why: Why,
  data: Data,
  twofive: TwoFive,
  model: Model,
  metric: Metric,
  curve: Curve,
  results: Results,
  insight: Insight,
  miss: Miss,
  demo: Demo,
  eng: Engineering,
  outro: ({ dur }) => (
    <SeriesOutro
      week={WEEK}
      dur={dur}
      repo="github.com/Fares-Ayman-1/gi-tract-segmentation"
      kaggle="kaggle.com/code/faresayman1/gi-tract-2-5d-u-net-and-the-competition-metric"
      scope="A research demo, not a medical device."
      credit="Data: UW-Madison GI Tract Image Segmentation (Kaggle), non-commercial use"
    />
  ),
};

export const Main: React.FC = () => <Timeline scenes={SCENES} registry={registry} />;
