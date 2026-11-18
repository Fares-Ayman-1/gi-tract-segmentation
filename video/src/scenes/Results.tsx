import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { NUM, ORGAN_NAME, SLICES, f2, f3, f4, type Organ } from "../data";
import { C, FONT } from "../kit/theme";
import { BarList, Eyebrow, FadeUp, Panel, Scene, Words, clamp, enter, progress, useLayout } from "../kit/ui";
import { Caption, Legend, ORGAN_FILL, SliceView, centroid } from "./Mri";

const PanelTitle: React.FC<{ children: React.ReactNode; u: number }> = ({ children, u }) => (
  <div style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted, marginBottom: 18 * u }}>{children}</div>
);

// ------------------------------------------------------------------------------- 8. results
export const Results: React.FC<{ dur: number }> = ({ dur }) => {
  const { u, wide, width } = useLayout();
  const pw = wide ? 760 * u : width - 120 * u;
  const organs: Organ[] = ["stomach", "large_bowel", "small_bowel"];
  const size = (wide ? 26 : 22) * u;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 34 : 18) * u, padding: 40 * u }}>
        <Eyebrow size={18 * u}>{`Results · ${NUM.validSessions} held-out sessions · original resolution`}</Eyebrow>
        <Words text={`${f4(NUM.score)} on unseen patients.`} at={4} size={(wide ? 76 : 54) * u} highlight={["unseen"]} maxWidth={width - 100 * u} />
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: (wide ? 34 : 14) * u, alignItems: "stretch" }}>
          <FadeUp at={14}>
            <Panel style={{ padding: `${24 * u}px ${30 * u}px`, width: pw, boxSizing: "border-box", height: "100%" }}>
              <PanelTitle u={u}>Competition score · higher is better</PanelTitle>
              <BarList
                at={28}
                width={pw - 60 * u}
                size={size}
                max={1}
                gap={(wide ? 30 : 16) * u}
                rows={[
                  { label: "Predict nothing", value: NUM.baseScore, display: f4(NUM.baseScore), note: `Dice ${f4(NUM.baseDice)} · HD ${f4(NUM.baseHausdorff)}` },
                  { label: "2.5D U-Net", value: NUM.score, display: f4(NUM.score), hero: true, note: `Dice ${f4(NUM.dice)} · HD ${f4(NUM.hausdorff)}` },
                ]}
              />
            </Panel>
          </FadeUp>
          <FadeUp at={40}>
            <Panel style={{ padding: `${24 * u}px ${30 * u}px`, width: pw, boxSizing: "border-box", height: "100%" }}>
              <PanelTitle u={u}>Per organ · score</PanelTitle>
              <BarList
                at={56}
                width={pw - 60 * u}
                size={size * 0.9}
                max={1}
                gap={(wide ? 22 : 12) * u}
                rows={organs.map((o) => ({
                  label: ORGAN_NAME[o],
                  value: NUM.perClass[o].score,
                  display: f4(NUM.perClass[o].score),
                  note: `Dice ${f4(NUM.perClass[o].dice)} · HD ${f4(NUM.perClass[o].hausdorff)}`,
                }))}
              />
            </Panel>
          </FadeUp>
        </div>
        <FadeUp at={96}>
          <Panel style={{ padding: `${14 * u}px ${24 * u}px`, borderLeft: `6px solid ${C.gold}`, maxWidth: (wide ? 1560 : 960) * u, boxSizing: "border-box" }}>
            <span style={{ fontFamily: FONT.body, fontSize: (wide ? 24 : 20) * u, color: C.muted, lineHeight: 1.4 }}>
              <b style={{ color: C.ink, fontWeight: 600 }}>{`Why "nothing" still gets ${f2(NUM.baseDice)} Dice: `}</b>
              most slices contain no organ, and an empty guess on an empty slice is perfect. Hausdorff is what separates a real model.
            </span>
          </Panel>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 9. one insight: volume_p995
export const Insight: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const S = SLICES.insight;
  const w = (wide ? 470 : 400) * u;
  const zoom = interpolate(f, [20, 70], [1, 2.1], { ...clamp, easing: (t) => 1 - Math.pow(1 - t, 3) });
  const origin = centroid(S.truth.stomach);
  const cards = [
    { k: "Normalised by its own 3 slices", v: NUM.p995Without, outlines: S.pred_slices, tone: C.miss },
    { k: "Normalised by the whole session", v: NUM.p995With, outlines: S.pred_volume, tone: C.gold },
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 28 : 18) * u, padding: 40 * u }}>
        <Eyebrow size={18 * u}>One insight · volume_p995</Eyebrow>
        <Words text="Normalise the way training did." at={4} size={(wide ? 70 : 48) * u} highlight={["training"]} maxWidth={width - 100 * u} />
        <div style={{ display: "flex", gap: (wide ? 40 : 16) * u }}>
          {cards.map((c, i) => (
            <FadeUp key={c.k} at={12 + i * 8}>
              <Panel style={{ padding: 14 * u, display: "flex", flexDirection: "column", gap: 12 * u, width: w + 28 * u, boxSizing: "border-box" }}>
                <SliceView
                  png={S.png}
                  size={S.size}
                  w={w}
                  zoom={zoom}
                  origin={origin}
                  radius={10 * u}
                  layers={[
                    { outlines: c.outlines, organs: ["stomach"], trace: progress(f, 70 + i * 10, 100 + i * 10), fill: progress(f, 90 + i * 10, 110 + i * 10) },
                    { outlines: S.truth, organs: ["stomach"], dashed: true, trace: enter(f, 60) },
                  ]}
                />
                <div style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 23 : 18) * u, color: C.ink }}>{c.k}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 12 * u, opacity: enter(f, 112 + i * 10) }}>
                  <span style={{ fontFamily: FONT.mono, fontSize: (wide ? 16 : 14) * u, color: C.muted, letterSpacing: "0.1em" }}>STOMACH DICE</span>
                  <span style={{ fontFamily: FONT.head, fontWeight: 700, fontSize: (wide ? 40 : 32) * u, color: C.ink, background: c.tone, borderRadius: 10 * u, padding: `${2 * u}px ${12 * u}px` }}>{f3(c.v)}</span>
                </div>
              </Panel>
            </FadeUp>
          ))}
        </div>
        <FadeUp at={60}>
          <Legend
            size={(wide ? 19 : 16) * u}
            organs={[]}
            extra={
              <div style={{ display: "flex", alignItems: "center", gap: 10 * u }}>
                <span style={{ width: 20 * u, height: 20 * u, borderRadius: 5 * u, background: ORGAN_FILL.stomach, border: `1.5px solid ${C.ink}` }} />
                <span style={{ fontFamily: FONT.body, fontWeight: 500, fontSize: (wide ? 19 : 16) * u, color: C.ink, marginRight: 18 * u }}>Predicted stomach</span>
                <svg width={40 * u} height={20 * u}>
                  <rect x={0} y={0} width={40 * u} height={20 * u} rx={4 * u} fill="#000" />
                  <line x1={4 * u} x2={36 * u} y1={10 * u} y2={10 * u} stroke="#fff" strokeWidth={2.5 * u} strokeDasharray={`${6 * u} ${4 * u}`} />
                </svg>
                <span style={{ fontFamily: FONT.body, fontWeight: 500, fontSize: (wide ? 19 : 16) * u, color: C.ink }}>Ground truth</span>
              </div>
            }
          />
        </FadeUp>
        <FadeUp at={130}>
          <Caption size={(wide ? 15 : 12) * u}>{`${S.case_day} · slice ${S.z} · source: README, "Why volume_p995 matters"`}</Caption>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 10. an honest miss
export const Miss: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const S = SLICES.miss;
  const w = (wide ? 470 : 400) * u;
  const show = progress(f, 20, 50);
  const ring = enter(f, 74);
  const zoom = interpolate(f, [80, 130], [1, 1.5], { ...clamp, easing: (t) => 1 - Math.pow(1 - t, 3) });
  const origin = centroid(S.truth.stomach);
  const cards = [
    { k: "Ground truth", layers: [{ outlines: S.truth, fill: show, trace: show }] },
    {
      k: "Model prediction",
      layers: [
        { outlines: S.pred, fill: show, trace: show },
        { outlines: S.truth, organs: ["stomach" as Organ], color: C.miss, fill: ring * 1.3, trace: ring, strokeW: 2.2 },
      ],
    },
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 28 : 18) * u, padding: 40 * u }}>
        <Eyebrow size={18 * u}>An honest miss</Eyebrow>
        <Words text="Where it still fails." at={4} size={(wide ? 70 : 50) * u} highlight={["fails."]} maxWidth={width - 100 * u} />
        <div style={{ display: "flex", gap: (wide ? 40 : 16) * u }}>
          {cards.map((c, i) => (
            <FadeUp key={c.k} at={10 + i * 8}>
              <Panel style={{ padding: 14 * u, display: "flex", flexDirection: "column", gap: 12 * u, width: w + 28 * u, boxSizing: "border-box" }}>
                <SliceView png={S.png} size={S.size} w={w} layers={c.layers} radius={10 * u} zoom={zoom} origin={origin} />
                <div style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 23 : 19) * u, color: C.ink }}>{c.k}</div>
              </Panel>
            </FadeUp>
          ))}
        </div>
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: (wide ? 30 : 12) * u, alignItems: "center" }}>
          <FadeUp at={40}>
            <Legend size={(wide ? 19 : 16) * u} />
          </FadeUp>
          <FadeUp at={86}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 * u }}>
              <span style={{ fontFamily: FONT.head, fontWeight: 700, fontSize: (wide ? 34 : 28) * u, color: C.ink, background: C.miss, borderRadius: 10 * u, padding: `${2 * u}px ${12 * u}px` }}>{`Stomach Dice ${f2(NUM.missDice)}`}</span>
              <span style={{ fontFamily: FONT.body, fontSize: (wide ? 22 : 18) * u, color: C.muted, maxWidth: (wide ? 560 : 520) * u, lineHeight: 1.35 }}>
                A small, low-contrast piece at the edge of the organ: not predicted at all.
              </span>
            </div>
          </FadeUp>
        </div>
        <FadeUp at={110}>
          <Caption size={(wide ? 15 : 12) * u}>{`${S.case_day} · slice ${S.z} · source: README, "An honest miss"`}</Caption>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

