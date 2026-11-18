import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { NUM, SLICES, type Organ } from "../data";
import { SERIES } from "../kit/series";
import { C, FONT, SHADOW } from "../kit/theme";
import { Body, Chip, Eyebrow, FadeUp, Panel, Scene, Words, enter, progress, useLayout } from "../kit/ui";
import { Caption, Legend, SliceView } from "./Mri";

const H = SLICES.hook;
const TRACE_ORDER: Organ[] = ["stomach", "small_bowel", "large_bowel"];

// ------------------------------------------------------------------------------- 1. hook + series title
export const Hook: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide } = useLayout();
  const s = SERIES[1];
  const imgW = (wide ? 700 : 520) * u;
  const layers = TRACE_ORDER.map((o, i) => ({
    outlines: H.pred,
    organs: [o],
    trace: progress(f, 14 + i * 16, 54 + i * 16),
    fill: progress(f, 50 + i * 16, 74 + i * 16),
  }));
  const card = (
    <FadeUp at={0} dist={30}>
      <Panel float style={{ padding: 18 * u, display: "flex", flexDirection: "column", gap: 14 * u, alignItems: "center" }}>
        <SliceView png={H.png} size={H.size} w={imgW} layers={layers} zoom={interpolate(f, [0, dur], [1.0, 1.06])} />
        <Legend at={60} size={(wide ? 22 : 19) * u} />
      </Panel>
    </FadeUp>
  );
  const title = (
    <div style={{ display: "flex", flexDirection: "column", gap: (wide ? 24 : 12) * u, alignItems: wide ? "flex-start" : "center", maxWidth: (wide ? 820 : 1000) * u }}>
      <Eyebrow at={4} size={(wide ? 20 : 16) * u}>Week 2 of 4 · ML portfolio</Eyebrow>
      <Words text={s.title} at={10} size={(wide ? 96 : 60) * u} align={wide ? "left" : "center"} highlight={["Segmentation"]} />
      <Words text={s.sub} at={24} size={(wide ? 42 : 30) * u} color={C.muted} lh={1.15} align={wide ? "left" : "center"} />
      <FadeUp at={70} style={{ marginTop: (wide ? 14 : 2) * u }}>
        <Caption size={(wide ? 16 : 13) * u}>{`Model prediction · ${H.case_day} · slice ${H.z} · patient never seen in training`}</Caption>
      </FadeUp>
    </div>
  );
  return (
    <Scene dur={dur} inF={1}>
      <AbsoluteFill style={{ flexDirection: wide ? "row" : "column", alignItems: "center", justifyContent: "center", gap: (wide ? 80 : 26) * u, padding: wide ? `0 ${110 * u}px` : `${36 * u}px` }}>
        {card}
        {title}
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 2. why
export const Why: React.FC<{ dur: number }> = ({ dur }) => {
  const { u, wide, width } = useLayout();
  const w = (wide ? 330 : 270) * u;
  const steps = [
    { k: "Every session", t: "An MRI scan of the abdomen", layers: [] },
    { k: "By hand", t: "Stomach and bowel outlined, so the beam can avoid them", layers: [{ outlines: H.truth, fill: 1 }] },
    { k: "This project", t: "A model draws the outlines automatically", layers: [{ outlines: H.pred, fill: 1 }] },
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 40 : 28) * u, padding: 50 * u }}>
        <Eyebrow size={18 * u}>Why · MR-guided radiotherapy</Eyebrow>
        <Words text="Outlined by hand, every session." at={4} size={(wide ? 76 : 54) * u} highlight={["hand,"]} maxWidth={width - 100 * u} />
        <div style={{ display: "flex", gap: (wide ? 34 : 18) * u, alignItems: "stretch" }}>
          {steps.map((s, i) => (
            <FadeUp key={s.k} at={24 + i * 12}>
              <Panel style={{ padding: 14 * u, width: w + 28 * u, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 12 * u, height: "100%" }}>
                <SliceView png={H.png} size={H.size} w={w} layers={s.layers} radius={10 * u} />
                <div style={{ display: "flex", alignItems: "center", gap: 10 * u }}>
                  <span style={{ width: 9 * u, height: 9 * u, borderRadius: 9, background: C.gold }} />
                  <span style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: (wide ? 16 : 14) * u, letterSpacing: "0.14em", textTransform: "uppercase", color: C.ink }}>{s.k}</span>
                </div>
                <div style={{ fontFamily: FONT.body, fontSize: (wide ? 23 : 19) * u, color: C.muted, lineHeight: 1.35 }}>{s.t}</div>
              </Panel>
            </FadeUp>
          ))}
        </div>
        <FadeUp at={70}>
          <Panel style={{ padding: `${12 * u}px ${24 * u}px`, borderLeft: `6px solid ${C.miss}`, display: "flex", gap: 14 * u, alignItems: "center" }}>
            <span style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 24 : 20) * u, color: C.ink }}>Research demo, not a medical device.</span>
            {wide && <span style={{ fontFamily: FONT.body, fontSize: 22 * u, color: C.muted }}>Not clinically validated.</span>}
          </Panel>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 3. data + patient split
export const Data: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const stats = [
    { v: NUM.sessions.toLocaleString("en-US"), l: "MRI sessions" },
    { v: NUM.patients.toLocaleString("en-US"), l: "patients" },
    { v: NUM.slices.toLocaleString("en-US"), l: "slices" },
  ];
  const trainP = NUM.patients - NUM.validPatients;
  const cols = wide ? 17 : 17;
  const dot = (wide ? 30 : 26) * u;
  const gap = (wide ? 12 : 10) * u;
  const split = progress(f, 80, 115);
  const dots = Array.from({ length: NUM.patients }, (_, i) => i);
  const gridW = cols * dot + (cols - 1) * gap;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 34 : 26) * u, padding: 50 * u }}>
        <Eyebrow size={18 * u}>The data · UW-Madison GI Tract (Kaggle)</Eyebrow>
        <div style={{ display: "flex", gap: (wide ? 70 : 34) * u }}>
          {stats.map((s, i) => (
            <FadeUp key={s.l} at={6 + i * 6}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (wide ? 104 : 72) * u, color: C.ink, letterSpacing: "-0.04em", lineHeight: 1 }}>{s.v}</div>
                <div style={{ fontFamily: FONT.body, fontSize: (wide ? 28 : 24) * u, color: C.muted, marginTop: 6 * u }}>{s.l}</div>
              </div>
            </FadeUp>
          ))}
        </div>
        <Words text="Validated on patients it never saw." at={50} size={(wide ? 58 : 44) * u} highlight={["never"]} maxWidth={width - 100 * u} />
        <FadeUp at={60}>
          <Panel style={{ padding: `${24 * u}px ${30 * u}px`, display: "flex", flexDirection: "column", gap: 16 * u }}>
            <div style={{ display: "flex", flexWrap: "wrap", width: gridW, gap }}>
              {dots.map((i) => {
                const val = i >= trainP;
                const o = enter(f, 60 + i * 0.4, 14);
                return (
                  <span
                    key={i}
                    style={{
                      width: dot,
                      height: dot,
                      borderRadius: dot,
                      opacity: o,
                      background: val && split > 0.5 ? C.gold : "rgba(12,23,51,0.75)",
                      boxShadow: val && split > 0.5 ? SHADOW.goldGlow : undefined,
                    }}
                  />
                );
              })}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 20 * u, opacity: split }}>
              <Legendish color="rgba(12,23,51,0.75)" u={u} wide={wide} text={`Training · ${trainP} patients, ${NUM.trainSessions} sessions`} />
              <Legendish color={C.gold} u={u} wide={wide} text={`Validation · ${NUM.validPatients} patients, ${NUM.validSessions} sessions`} />
            </div>
          </Panel>
        </FadeUp>
        <FadeUp at={100}>
          <Chip size={(wide ? 19 : 16) * u}>5-fold GroupKFold by patient · no patient in both</Chip>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

const Legendish: React.FC<{ color: string; text: string; u: number; wide: boolean }> = ({ color, text, u, wide }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10 * u }}>
    <span style={{ width: 16 * u, height: 16 * u, borderRadius: 16, background: color }} />
    <span style={{ fontFamily: FONT.body, fontWeight: 500, fontSize: (wide ? 22 : 18) * u, color: C.ink }}>{text}</span>
  </div>
);

// ------------------------------------------------------------------------------- 4. 2.5D input
export const TwoFive: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const w = (wide ? 300 : 260) * u;
  const gap = (wide ? 40 : 24) * u;
  const p = progress(f, 88, 124);
  const tensor = enter(f, 120);
  const labels = ["z − 2", "z", "z + 2"];
  const stackShift = wide ? -280 * u : 0;
  const off = 34 * u;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 36 : 26) * u, padding: 50 * u }}>
        <Eyebrow size={18 * u}>2.5D input</Eyebrow>
        <Words text="Three slices become one input." at={4} size={(wide ? 72 : 52) * u} highlight={["Three"]} maxWidth={width - 100 * u} />
        <div style={{ position: "relative", width: width - 100 * u, height: w + 90 * u }}>
          {H.context.map((png, i) => {
            const rowX = (i - 1) * (w + 24 * u + gap);
            const stX = stackShift + (1 - i) * off; // z - 2 in front (bottom-left), z + 2 at the back
            const stY = (i - 1) * -off * 0.8;
            const x = interpolate(p, [0, 1], [rowX, stX]);
            const y = interpolate(p, [0, 1], [0, stY]);
            const s = enter(f, 20 + i * 6);
            return (
              <div
                key={png}
                style={{
                  position: "absolute",
                  left: "50%",
                  top: 0,
                  transform: `translate(${x - w / 2 - 12 * u}px, ${y + (1 - s) * 30}px)`,
                  opacity: s,
                  zIndex: 3 - i,
                }}
              >
                <Panel style={{ padding: 12 * u, display: "flex", flexDirection: "column", gap: 10 * u, alignItems: "center" }}>
                  <SliceView png={png} size={H.size} w={w} radius={10 * u} />
                  <div style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: (wide ? 17 : 14) * u, color: C.ink, letterSpacing: "0.06em" }}>
                    {`CH ${i + 1} · ${labels[i]} · SLICE ${H.context_z[i]}`}
                  </div>
                </Panel>
              </div>
            );
          })}
          {wide && (
            <div style={{ position: "absolute", left: "50%", top: w / 2 - 50 * u, transform: `translateX(${stackShift + w / 2 + 90 * u}px)`, opacity: tensor, display: "flex", alignItems: "center", gap: 26 * u }}>
              <svg width={90 * u} height={30 * u} viewBox="0 0 90 30">
                <path d="M0 15 H78 M66 4 L82 15 L66 26" stroke={C.ink} strokeWidth={3} fill="none" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <TensorCard u={u} />
            </div>
          )}
        </div>
        {!wide && (
          <div style={{ opacity: tensor, marginTop: 6 * u }}>
            <TensorCard u={u} />
          </div>
        )}
        <Body at={40} size={wide ? 28 : 24} align="center">
          Neighbouring slices give a 2D network depth context, almost for free.
        </Body>
      </AbsoluteFill>
    </Scene>
  );
};

const TensorCard: React.FC<{ u: number }> = ({ u }) => (
  <Panel style={{ padding: `${18 * u}px ${26 * u}px`, borderLeft: `6px solid ${C.gold}`, display: "flex", flexDirection: "column", gap: 6 * u }}>
    <div style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: 15 * u, letterSpacing: "0.16em", color: C.muted }}>INPUT TENSOR</div>
    <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 44 * u, color: C.ink, letterSpacing: "-0.03em" }}>{`3 × ${NUM.size} × ${NUM.size}`}</div>
    <div style={{ fontFamily: FONT.body, fontSize: 20 * u, color: C.muted }}>{`stride ${NUM.stride} · one slice, two neighbours`}</div>
  </Panel>
);
