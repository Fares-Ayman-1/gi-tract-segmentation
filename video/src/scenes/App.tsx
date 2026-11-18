import React from "react";
import { AbsoluteFill, OffthreadVideo, interpolate, staticFile, useCurrentFrame } from "remotion";
import { NUM, f3 } from "../data";
import { hasFile } from "../kit/Timeline";
import { C, FONT } from "../kit/theme";
import { BrowserFrame, Chip, Eyebrow, FadeUp, Panel, Scene, Words, enter, progress, useLayout } from "../kit/ui";

// public/demo.mp4: the real Streamlit viewer (app/viewer.py), screen-recorded by scripts/capture.mjs from
// capture/plan.json. 1440 x 1080, 17.6 s: case108_day0 from slice 78 (the miss) to 73, then case108_day13,
// slices 91 -> 95 -> 91. Played at RATE so the whole recording fits the scene.
const CAP_W = 1440;
const CAP_H = 1080;
const RATE = 1.58;

export const Demo: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const fw = wide ? 1060 * u : width - 140 * u;
  const pad = 16 * u;
  const vw = fw - 2 * pad - 2;
  const vh = (vw * CAP_H) / CAP_W;
  const e = progress(f, 0, 30);
  const t = (f * RATE) / 30; // seconds into the recording
  const step = t < 4.6 ? 0 : t < 8 ? 1 : 2;
  const steps = ["case108_day0 · slice 78 → 73", "switching session", "case108_day13 · slice 91 → 95 → 91"];
  const frame = (
    <div style={{ opacity: e, transform: `translateY(${(1 - e) * 40}px)` }}>
      <BrowserFrame url="localhost:8502 · app/viewer.py" width={fw}>
        <div style={{ width: vw, height: vh, background: "#fff", overflow: "hidden" }}>
          {hasFile("demo.mp4") ? (
            <OffthreadVideo src={staticFile("demo.mp4")} muted playbackRate={RATE} style={{ width: vw, height: vh, transform: `scale(${interpolate(f, [0, dur], [1, 1.03])})`, transformOrigin: "50% 70%" }} />
          ) : (
            <div style={{ width: vw, height: vh, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FONT.mono, color: C.muted }}>public/demo.mp4 missing</div>
          )}
        </div>
      </BrowserFrame>
    </div>
  );
  const live = (
    <div style={{ display: "flex", alignItems: "center", gap: 10 * u, opacity: enter(f, 20) }}>
      <span style={{ width: 11 * u, height: 11 * u, borderRadius: 11, background: C.gold }} />
      <span style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: (wide ? 17 : 15) * u, color: C.ink, letterSpacing: "0.08em" }}>{steps[step]}</span>
    </div>
  );
  if (!wide) {
    return (
      <Scene dur={dur}>
        <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 18 * u, padding: 30 * u }}>
          <Eyebrow size={16 * u}>The real app · Streamlit viewer</Eyebrow>
          {frame}
          {live}
          <div style={{ display: "flex", gap: 10 * u, flexWrap: "wrap", justifyContent: "center" }}>
            <Chip at={40} size={16 * u}>model.onnx on CPU</Chip>
            <Chip at={46} size={16 * u}>{`${NUM.validSessions} held-out sessions`}</Chip>
            <Chip at={52} size={16 * u}>truth vs prediction</Chip>
          </div>
        </AbsoluteFill>
      </Scene>
    );
  }
  const points = [
    { k: "ONNX", t: "model.onnx on CPU, ONNX Runtime" },
    { k: String(NUM.validSessions), t: "held-out sessions, patients never seen in training" },
    { k: "3", t: "panels: scan, ground truth, model prediction" },
    { k: "78", t: "the session opens on the miss: case108_day0" },
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 60 * u, padding: `0 ${70 * u}px` }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 * u }}>
          {frame}
          {live}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 24 * u, width: 560 * u }}>
          <Eyebrow size={18 * u} at={8}>The real app · Streamlit viewer</Eyebrow>
          <Words text="Scroll any held-out session." at={14} size={60 * u} align="left" highlight={["held-out"]} />
          <div style={{ display: "flex", flexDirection: "column", gap: 16 * u, marginTop: 8 * u }}>
            {points.map((p, i) => (
              <FadeUp key={p.k} at={44 + i * 10}>
                <div style={{ display: "flex", alignItems: "center", gap: 16 * u }}>
                  <div style={{ minWidth: 70 * u, height: 46 * u, borderRadius: 12 * u, background: C.goldSoft, border: `1px solid ${C.gold}`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FONT.mono, fontWeight: 700, fontSize: 17 * u, color: C.ink }}>
                    {p.k}
                  </div>
                  <div style={{ fontFamily: FONT.body, fontSize: 24 * u, color: C.muted, lineHeight: 1.35 }}>{p.t}</div>
                </div>
              </FadeUp>
            ))}
          </div>
        </div>
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 12. engineering
export const Engineering: React.FC<{ dur: number }> = ({ dur }) => {
  const { u, wide, width } = useLayout();
  const cols = wide ? 3 : 2;
  const gap = (wide ? 22 : 16) * u;
  const cw = wide ? 500 * u : (width - 100 * u - gap) / 2;
  const cards = [
    { k: "ONNX", t: "One model.onnx file, shared by the API and the viewer" },
    { k: "FastAPI", t: "POST /v1/segmentations, no PyTorch at serving time" },
    { k: "Docker", t: "API + viewer image with ONNX Runtime only" },
    { k: "CI", t: "GitHub Actions: ruff, mypy, pytest, Docker build" },
    {
      k: "Drift monitoring",
      t: `PSI ${f3(NUM.psi)} · unseen sizes ${Math.round(NUM.unseen * 100)}% · prevalence Δ ${f3(NUM.prevalence)} → ${NUM.retrain ? "retrain" : "no retrain"}`,
    },
    { k: "MLflow", t: `Every run tracked, metrics and artefacts · run ${NUM.runIds.model.slice(0, 8)}` },
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 40 : 26) * u, padding: 50 * u }}>
        <Eyebrow size={18 * u}>Engineering</Eyebrow>
        <Words text="Built to run, not just to train." at={4} size={(wide ? 72 : 50) * u} highlight={["run,"]} maxWidth={width - 100 * u} />
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, ${cw}px)`, gap }}>
          {cards.map((c, i) => (
            <FadeUp key={c.k} at={22 + i * 5}>
              <Panel style={{ padding: `${20 * u}px ${24 * u}px`, height: (wide ? 150 : 150) * u, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 10 * u, borderTop: `5px solid ${C.gold}` }}>
                <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (wide ? 30 : 26) * u, color: C.ink, letterSpacing: "-0.02em" }}>{c.k}</div>
                <div style={{ fontFamily: FONT.body, fontSize: (wide ? 21 : 18) * u, color: C.muted, lineHeight: 1.35 }}>{c.t}</div>
              </Panel>
            </FadeUp>
          ))}
        </div>
      </AbsoluteFill>
    </Scene>
  );
};
