# Project video

A 92-second explainer (Week 2 of the 4-week ML portfolio series) built with [Remotion](https://www.remotion.dev/)
(React → MP4) and the shared Quantro AI light-theme kit in `src/kit/` (no logo). Every MRI slice on screen is a
real validation slice with its real ground truth, every prediction is `models/model.onnx` run through the
project's own inference code, and every number is read from `reports/metrics.json`, `reports/monitoring.json`
and `models/model.json` (via `src/data.ts`), or is a README number cited there.

| Output | Size | Use |
|---|---|---|
| `out/gi-wide.mp4` | 1920 × 1080 | YouTube, GitHub README, portfolio site |
| `out/gi-square.mp4` | 1080 × 1080 | LinkedIn feed |

## Rebuild

```bash
# from the project root (needs data/cache, data/raw and models/; CPU only, about a minute)
.venv/Scripts/python video/scripts/extract.py   # -> video/public/data/*.png + slices.json

cd video
npm install
npm run studio          # live preview in the browser
node scripts/stills.mjs Wide 170 1580 2740      # review stills -> out/stills/
npm run render:wide     # -> out/gi-wide.mp4
npm run render:square   # -> out/gi-square.mp4
```

`public/data/` (5 slices + outlines), `public/demo.mp4` and `capture/` are derived from the competition data,
so they are git-ignored. Only a handful of slices are ever on screen (`D:\ml-portfolio\DATA_RULES.md`).

`scripts/extract.py` exports:
- `case117_day16`, slices 34 / 36 / 38: the hook (36) and the 2.5D input, from the 256 px cache, predicted
  exactly like the viewer (cached slices / 255 into the ONNX model, through `Segmenter.predict`);
- `case108_day0`, slice 73: raw PNGs through the API's `Segmenter.predict` with and without `volume_p995`
  (reproduces the README's stomach Dice 0.245 → 0.604);
- `case108_day0`, slice 78: the honest miss (stomach Dice 0.00, as in the README).

## Demo recording, music and sound

```bash
# real Streamlit viewer, screen-recorded in headless Chrome (light colour scheme)
cd ..   # project root
UV_CACHE_DIR=D:/uv-cache uv run --no-sync streamlit run app/viewer.py --server.port 8502 --server.headless true &
cd video
node scripts/capture.mjs capture/plan.json      # -> capture/demo.webm
ffmpeg -i capture/demo.webm -vf "fps=30,format=yuv420p" -c:v libx264 -crf 14 public/demo.mp4
# stop the server afterwards (netstat -ano | findstr :8502, then taskkill //PID <pid> //F)

# original music bed + sound effects, timed to scenes.json (D minor, 88 BPM)
cd ..; .venv/Scripts/python video/scripts/music.py --root D --mode minor --bpm 88
```

- `public/demo.mp4`: 17.6 s of the real viewer at 1440 × 1080. It opens on `case108_day0` slice 78 (the miss),
  steps to 73, switches to `case108_day13` and scrolls 91 → 95 → 91. The first open of a session runs the
  model on the whole volume (about 20 s on CPU), so warm the cache by opening both sessions once before
  recording. The demo scene plays it at 1.58× (`RATE` in `src/scenes/App.tsx`).
- `public/music.mp3`, `public/sfx.mp3`: synthesized from the cues in `scenes.json` (energy per scene, ticks
  when bars land, a riser into the results). Re-run `music.py` after changing any `dur`.

The video uses whichever of these files exist in `public/`.

## Scenes

`scenes.json` is the single source of truth for order and duration (30 fps, 10-frame cross-fades):
hook + series title · why (radiotherapy) · data and patient split · 2.5D input · the model · the metric ·
training curve · results · one insight (volume_p995) · an honest miss · the real app · engineering · outro.

| File | Scenes |
|---|---|
| `src/scenes/Intro.tsx` | hook, why, data, twofive |
| `src/scenes/Model.tsx` | model, metric, curve |
| `src/scenes/Results.tsx` | results, insight, miss |
| `src/scenes/App.tsx` | demo, eng |
| `src/scenes/Mri.tsx` | `SliceView` (slice + SVG organ outlines), legend |
