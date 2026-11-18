import React from "react";
import { Img, staticFile } from "remotion";
import { ORGANS, ORGAN_NAME, type Organ, type Outlines, type Poly } from "../data";
import { C, FONT } from "../kit/theme";
import { FadeUp } from "../kit/ui";

// Data-illustration colours (same order as the viewer: large bowel gold, small bowel blue, stomach green)
export const ORGAN_FILL: Record<Organ, string> = { large_bowel: C.gold, small_bowel: C.blue, stomach: C.green };

const toPath = (p: Poly) => p.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ") + " Z";

export type Layer = {
  outlines: Outlines;
  organs?: Organ[];
  /** 0..1: how much of each outline is traced */
  trace?: number;
  /** 0..1: fill opacity multiplier */
  fill?: number;
  /** dashed navy outline, no fill (ground truth reference) */
  dashed?: boolean;
  /** override fill colour (e.g. C.miss) */
  color?: string;
  strokeW?: number;
};

/** One real MRI slice (grayscale PNG) with organ outlines drawn as SVG in image-pixel coordinates. */
export const SliceView: React.FC<{ png: string; size: number; w: number; layers?: Layer[]; radius?: number; zoom?: number; origin?: number[] }> = ({
  png,
  size,
  w,
  layers = [],
  radius = 14,
  zoom = 1,
  origin,
}) => {
  const k = size / w; // image px per display px
  return (
    <div style={{ width: w, height: w, position: "relative", borderRadius: radius, overflow: "hidden", background: "#000", flexShrink: 0 }}>
      <div style={{ position: "absolute", inset: 0, transform: `scale(${zoom})`, transformOrigin: origin ? `${(origin[0] / size) * 100}% ${(origin[1] / size) * 100}%` : "50% 50%" }}>
        <Img src={staticFile(png)} style={{ width: w, height: w, display: "block" }} />
        <svg viewBox={`0 0 ${size} ${size}`} width={w} height={w} style={{ position: "absolute", left: 0, top: 0 }}>
          {layers.map((L, li) =>
            (L.organs ?? ORGANS).map((o) =>
              L.outlines[o].map((p, pi) => {
                const t = L.trace ?? 1;
                const fillO = (L.fill ?? 1) * (L.dashed ? 0 : 0.62);
                return (
                  <path
                    key={`${li}-${o}-${pi}`}
                    d={toPath(p)}
                    pathLength={L.dashed ? undefined : 1}
                    fill={L.color ?? ORGAN_FILL[o]}
                    fillOpacity={fillO}
                    stroke={L.dashed ? "#FFFFFF" : C.ink}
                    strokeWidth={((L.strokeW ?? (L.dashed ? 2 : 1.6)) * k) / zoom}
                    strokeLinejoin="round"
                    strokeDasharray={L.dashed ? `${(5 * k) / zoom} ${(3.5 * k) / zoom}` : "1 1"}
                    strokeDashoffset={L.dashed ? 0 : 1 - t}
                    opacity={L.dashed ? t : 1}
                  />
                );
              }),
            ),
          )}
        </svg>
      </div>
    </div>
  );
};

/** Centre of an organ's outlines (image px), for zooming in. */
export const centroid = (polys: Poly[]): number[] => {
  const pts = polys.flat();
  const n = Math.max(1, pts.length);
  return [pts.reduce((a, p) => a + p[0], 0) / n, pts.reduce((a, p) => a + p[1], 0) / n];
};

/** Swatch legend for the three organs. */
export const Legend: React.FC<{ at?: number; size?: number; dir?: "row" | "column"; organs?: Organ[]; extra?: React.ReactNode }> = ({
  at = 0,
  size = 22,
  dir = "row",
  organs = ORGANS,
  extra,
}) => (
  <div style={{ display: "flex", flexDirection: dir, gap: dir === "row" ? size * 1.2 : size * 0.6, flexWrap: "wrap", justifyContent: "center" }}>
    {organs.map((o, i) => (
      <FadeUp key={o} at={at + i * 4} dist={10}>
        <div style={{ display: "flex", alignItems: "center", gap: size * 0.5 }}>
          <span style={{ width: size * 0.9, height: size * 0.9, borderRadius: size * 0.22, background: ORGAN_FILL[o], border: `1.5px solid ${C.ink}` }} />
          <span style={{ fontFamily: FONT.body, fontWeight: 500, fontSize: size, color: C.ink }}>{ORGAN_NAME[o]}</span>
        </div>
      </FadeUp>
    ))}
    {extra}
  </div>
);

/** Mono caption under an image. */
export const Caption: React.FC<{ children: React.ReactNode; size?: number }> = ({ children, size = 16 }) => (
  <div style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: size, letterSpacing: "0.12em", textTransform: "uppercase", color: C.muted }}>{children}</div>
);
