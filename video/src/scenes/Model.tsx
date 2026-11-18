import React from "react";
import { AbsoluteFill, staticFile, useCurrentFrame } from "remotion";
import { NUM, SLICES, f3, f4 } from "../data";
import { C, FONT } from "../kit/theme";
import { Chip, Eyebrow, FadeUp, Panel, Scene, Words, enter, progress, useLayout } from "../kit/ui";
import { ORGAN_FILL } from "./Mri";

// ------------------------------------------------------------------------------- 5. the model
export const Model: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const W = 1000;
  const Hh = 470;
  const lv = [0, 1, 2, 3, 4];
  const cy = (i: number) => 150 + i * 62;
  const bh = (i: number) => 230 - i * 42;
  const ex = (i: number) => 200 + i * 78;
  const dx = (i: number) => 800 - i * 78;
  const svgW = (wide ? 1180 : 940) * u;
  const facts = [
    { k: "Encoder", v: "ResNet-34 · ImageNet weights" },
    { k: "Training", v: `${NUM.epochs} epochs · mixed precision` },
    { k: "Time", v: `${Math.round(NUM.minutes)} minutes · ${NUM.gpu}` },
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 30 : 22) * u, padding: 50 * u }}>
        <Eyebrow size={18 * u}>The model</Eyebrow>
        <Words text="A U-Net with a pretrained encoder." at={4} size={(wide ? 70 : 50) * u} highlight={["pretrained"]} maxWidth={width - 100 * u} />
        <FadeUp at={10}>
          <Panel style={{ padding: 18 * u }}>
            <svg viewBox={`0 0 ${W} ${Hh}`} width={svgW} height={(svgW * Hh) / W}>
              {/* input: the three stacked slices */}
              {[2, 1, 0].map((i) => (
                <g key={i} opacity={enter(f, 12)}>
                  <image href={staticFile(SLICES.hook.context[i])} x={20 + i * 12} y={110 - i * 12} width={120} height={120} />
                  <rect x={20 + i * 12} y={110 - i * 12} width={120} height={120} fill="none" stroke={C.line} strokeWidth={2} />
                </g>
              ))}
              <text x={86} y={290} textAnchor="middle" fontFamily={FONT.mono} fontSize={17} fill={C.muted}>{`3 × ${NUM.size}²`}</text>
              {/* skips */}
              {lv.slice(0, 4).map((i) => {
                const p = progress(f, 56 + i * 4, 80 + i * 4);
                const x1 = ex(i) + 22;
                const x2 = dx(i) - 22;
                return <line key={`s${i}`} x1={x1} y1={cy(i)} x2={x1 + (x2 - x1) * p} y2={cy(i)} stroke={C.gold} strokeWidth={4} strokeDasharray="10 8" strokeLinecap="round" />;
              })}
              {/* encoder (navy) and decoder (white) */}
              {lv.map((i) => {
                const a = enter(f, 16 + i * 5, 18);
                const b = enter(f, 40 + (4 - i) * 5, 18);
                return (
                  <g key={i}>
                    <rect x={ex(i) - 22} y={cy(i) - bh(i) / 2} width={44} height={bh(i)} rx={8} fill={C.ink} opacity={a} />
                    {i < 4 && <rect x={dx(i) - 22} y={cy(i) - bh(i) / 2} width={44} height={bh(i)} rx={8} fill={C.card} stroke={C.ink} strokeWidth={3} opacity={b} />}
                  </g>
                );
              })}
              <text x={ex(2)} y={440} textAnchor="middle" fontFamily={FONT.mono} fontSize={17} fill={C.ink} opacity={enter(f, 30)} letterSpacing="2">
                ENCODER · RESNET-34
              </text>
              <text x={dx(2)} y={440} textAnchor="middle" fontFamily={FONT.mono} fontSize={17} fill={C.ink} opacity={enter(f, 60)} letterSpacing="2">
                DECODER
              </text>
              {/* output: three organ masks */}
              {(["large_bowel", "small_bowel", "stomach"] as const).map((o, i) => (
                <g key={o} opacity={enter(f, 72 + i * 4)}>
                  <rect x={870 + i * 12} y={110 - i * 12 + 0} width={100} height={100} rx={10} fill={ORGAN_FILL[o]} stroke={C.ink} strokeWidth={2} />
                </g>
              ))}
              <text x={932} y={270} textAnchor="middle" fontFamily={FONT.mono} fontSize={17} fill={C.muted} opacity={enter(f, 80)}>
                3 masks
              </text>
            </svg>
          </Panel>
        </FadeUp>
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: (wide ? 20 : 10) * u, alignItems: "center" }}>
          {facts.map((x, i) => (
            <FadeUp key={x.k} at={70 + i * 8}>
              <Panel style={{ padding: `${12 * u}px ${20 * u}px`, display: "flex", gap: 14 * u, alignItems: "baseline", borderLeft: `5px solid ${C.gold}` }}>
                <span style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted }}>{x.k}</span>
                <span style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 22 : 21) * u, color: C.ink }}>{x.v}</span>
              </Panel>
            </FadeUp>
          ))}
        </div>
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 6. the metric, drawn
export const Metric: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const ill = (wide ? 300 : 230) * u;
  const rows = [
    { k: "2.5D U-Net", d: NUM.dice, h: NUM.hausdorff, s: NUM.score, hero: true },
    { k: "Predict nothing", d: NUM.baseDice, h: NUM.baseHausdorff, s: NUM.baseScore, hero: false },
  ];
  const fs = (wide ? 30 : 22) * u;
  const hd = progress(f, 60, 84);
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 30 : 22) * u, padding: 50 * u }}>
        <Eyebrow size={18 * u}>The competition metric · re-implemented, unit-tested</Eyebrow>
        <Words text="Overlap, plus the worst distance." at={4} size={(wide ? 68 : 48) * u} highlight={["worst"]} maxWidth={width - 100 * u} />
        <FadeUp at={16}>
          <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (wide ? 50 : 34) * u, color: C.ink, letterSpacing: "-0.02em", textAlign: "center" }}>
            0.4 · Dice + 0.6 · (1 − Hausdorff)
          </div>
        </FadeUp>
        <div style={{ display: "flex", gap: (wide ? 40 : 18) * u }}>
          <FadeUp at={30}>
            <Panel style={{ padding: 16 * u, display: "flex", flexDirection: "column", alignItems: "center", gap: 8 * u, width: ill + 32 * u, boxSizing: "border-box" }}>
              <svg viewBox="0 0 300 170" width={ill} height={(ill * 170) / 300}>
                <ellipse cx={125} cy={85} rx={78} ry={62} fill={C.gold} fillOpacity={0.6} stroke={C.ink} strokeWidth={2.5} />
                <ellipse cx={180} cy={90} rx={74} ry={58} fill={C.blue} fillOpacity={0.55 * enter(f, 38)} stroke={C.ink} strokeWidth={2.5} strokeDasharray="6 5" opacity={enter(f, 38)} />
              </svg>
              <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 26 * u, color: C.ink }}>Dice</div>
              <div style={{ fontFamily: FONT.body, fontSize: 19 * u, color: C.muted, textAlign: "center" }}>overlap per slice and organ</div>
            </Panel>
          </FadeUp>
          <FadeUp at={44}>
            <Panel style={{ padding: 16 * u, display: "flex", flexDirection: "column", alignItems: "center", gap: 8 * u, width: ill + 32 * u, boxSizing: "border-box" }}>
              <svg viewBox="0 0 300 170" width={ill} height={(ill * 170) / 300}>
                <path d="M40 120 C60 40, 140 30, 170 70 C190 95, 160 140, 110 140 C80 140, 50 140, 40 120 Z" fill={C.gold} fillOpacity={0.6} stroke={C.ink} strokeWidth={2.5} />
                <path d="M70 110 C80 60, 130 55, 150 80 C165 100, 140 125, 110 125 C90 125, 75 125, 70 110 Z M215 52 a22 18 0 1 0 0.1 0 Z" fill={C.blue} fillOpacity={0.55} stroke={C.ink} strokeWidth={2.5} strokeDasharray="6 5" />
                <line x1={170} y1={70} x2={170 + (215 - 170) * hd} y2={70 + (34 - 70) * hd} stroke={C.ink} strokeWidth={3.5} strokeLinecap="round" />
                <circle cx={170} cy={70} r={5} fill={C.ink} opacity={hd} />
                <circle cx={215} cy={34} r={5} fill={C.ink} opacity={hd} />
              </svg>
              <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 26 * u, color: C.ink }}>Hausdorff</div>
              <div style={{ fontFamily: FONT.body, fontSize: 19 * u, color: C.muted, textAlign: "center" }}>farthest stray point, in 3D, ÷ diagonal</div>
            </Panel>
          </FadeUp>
        </div>
        <FadeUp at={92}>
          <Panel style={{ padding: `${16 * u}px ${26 * u}px`, display: "flex", flexDirection: "column", gap: 10 * u }}>
            {rows.map((r) => (
              <div key={r.k} style={{ display: "flex", alignItems: "baseline", gap: 16 * u, fontVariantNumeric: "tabular-nums" }}>
                <span style={{ width: (wide ? 230 : 150) * u, fontFamily: FONT.body, fontWeight: 600, fontSize: fs * 0.78, color: r.hero ? C.ink : C.muted }}>{r.k}</span>
                <span style={{ fontFamily: FONT.mono, fontSize: fs * 0.72, color: C.ink }}>{`0.4 × ${f4(r.d)} + 0.6 × (1 − ${f4(r.h)}) =`}</span>
                <span style={{ fontFamily: FONT.head, fontWeight: 700, fontSize: fs * 1.05, color: C.ink, background: r.hero ? C.goldSoft : undefined, padding: `0 ${8 * u}px`, borderRadius: 8 * u }}>{f4(r.s)}</span>
              </div>
            ))}
          </Panel>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

// ------------------------------------------------------------------------------- 7. training curve
type Pt = { x: number; y: number };
const Chart: React.FC<{
  w: number;
  h: number;
  vals: number[];
  lo: number;
  hi: number;
  ticks: number[];
  color: string;
  title: string;
  at: number;
  fmt: (v: number) => string;
  marks?: { i: number; label: string; tone: "miss" | "gold" }[];
}> = ({ w, h, vals, lo, hi, ticks, color, title, at, fmt, marks = [] }) => {
  const f = useCurrentFrame();
  const { u } = useLayout();
  const padL = 70 * u;
  const padB = 44 * u;
  const padT = 50 * u;
  const padR = 24 * u;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const pts: Pt[] = vals.map((v, i) => ({ x: padL + (i / (vals.length - 1)) * pw, y: padT + (1 - (v - lo) / (hi - lo)) * ph }));
  const p = progress(f, at, at + 60);
  const reach = p * (vals.length - 1);
  const d = pts.map((q, i) => `${i ? "L" : "M"}${q.x} ${q.y}`).join(" ");
  return (
    <svg width={w} height={h}>
      <text x={padL} y={26 * u} fontFamily={FONT.mono} fontSize={16 * u} fill={C.ink} letterSpacing={2 * u}>
        {title}
      </text>
      {ticks.map((t) => {
        const y = padT + (1 - (t - lo) / (hi - lo)) * ph;
        return (
          <g key={t}>
            <line x1={padL} x2={w - padR} y1={y} y2={y} stroke={C.line} strokeWidth={1} />
            <text x={padL - 10 * u} y={y + 5 * u} textAnchor="end" fontFamily={FONT.mono} fontSize={14 * u} fill={C.muted}>
              {fmt(t)}
            </text>
          </g>
        );
      })}
      {[1, 5, 10, 15].map((e) => (
        <text key={e} x={pts[e - 1].x} y={h - 14 * u} textAnchor="middle" fontFamily={FONT.mono} fontSize={14 * u} fill={C.muted}>
          {e}
        </text>
      ))}
      <defs>
        <clipPath id={`clip-${title}`}>
          <rect x={0} y={0} width={padL + p * pw + 2} height={h} />
        </clipPath>
      </defs>
      <path d={d} fill="none" stroke={color} strokeWidth={4 * u} strokeLinejoin="round" strokeLinecap="round" clipPath={`url(#clip-${title})`} />
      {pts.map((q, i) => (
        <circle key={i} cx={q.x} cy={q.y} r={5 * u} fill={C.card} stroke={color} strokeWidth={2.5 * u} opacity={reach >= i ? 1 : 0} />
      ))}
      {marks.map((m) => {
        const q = pts[m.i];
        const o = enter(f, at + 60 * (m.i / (vals.length - 1)) + 6, 16);
        const below = m.tone === "miss";
        const tw = m.label.length * 9.4 * u + 24 * u;
        const bx = Math.min(Math.max(q.x - tw / 2, padL), w - padR - tw);
        const by = below ? q.y + 16 * u : q.y - 50 * u;
        return (
          <g key={m.i} opacity={o}>
            <circle cx={q.x} cy={q.y} r={9 * u} fill={m.tone === "miss" ? C.miss : C.gold} stroke={C.ink} strokeWidth={2 * u} />
            <rect x={bx} y={by} width={tw} height={34 * u} rx={17 * u} fill={m.tone === "miss" ? C.miss : C.gold} />
            <text x={bx + tw / 2} y={by + 23 * u} textAnchor="middle" fontFamily={FONT.mono} fontSize={15 * u} fill={C.ink}>
              {m.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

export const Curve: React.FC<{ dur: number }> = ({ dur }) => {
  const { u, wide, width } = useLayout();
  const hist = NUM.history;
  const cw = wide ? 780 * u : width - 120 * u;
  const ch = wide ? 470 * u : 300 * u;
  const dipI = NUM.dip.epoch - 1;
  const bestI = NUM.best.epoch - 1;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 30 : 18) * u, padding: 40 * u }}>
        <Eyebrow size={18 * u}>{`Training · ${NUM.epochs} epochs · reports/metrics.json`}</Eyebrow>
        <Words text="Not a straight line up." at={4} size={(wide ? 70 : 50) * u} highlight={["straight"]} maxWidth={width - 100 * u} />
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: (wide ? 30 : 12) * u }}>
          <FadeUp at={12}>
            <Panel style={{ padding: 8 * u }}>
              <Chart w={cw} h={ch} vals={hist.map((h) => h.train_loss)} lo={0} hi={0.8} ticks={[0, 0.2, 0.4, 0.6, 0.8]} color={C.ink} title="TRAIN LOSS" at={20} fmt={(v) => v.toFixed(1)} />
            </Panel>
          </FadeUp>
          <FadeUp at={18}>
            <Panel style={{ padding: 8 * u }}>
              <Chart
                w={cw}
                h={ch}
                vals={hist.map((h) => h.val_dice_256)}
                lo={0.82}
                hi={0.93}
                ticks={[0.84, 0.88, 0.92]}
                color={C.gold}
                title="VALIDATION DICE · 256 PX"
                at={30}
                fmt={(v) => v.toFixed(2)}
                marks={[
                  { i: dipI, label: `epoch ${NUM.dip.epoch} dip · ${f3(NUM.dip.val_dice_256)}`, tone: "miss" },
                  { i: bestI, label: `best · epoch ${NUM.best.epoch} · ${f3(NUM.best.val_dice_256)}`, tone: "gold" },
                ]}
              />
            </Panel>
          </FadeUp>
        </div>
        <FadeUp at={100}>
          <Chip size={(wide ? 19 : 15) * u}>Per-epoch check at 256 px · the final score is measured at original resolution</Chip>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};
