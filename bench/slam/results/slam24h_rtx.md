# slam2 vs RTAB-Map — "24 h" life benchmark on i777 (RTX), stopped early at ~6.2–6.7 sim-h

Harness: `bench/slam/life24.py --hours 24 --rtab-param Rtabmap/TimeThr=700 --rtab-param Mem/BinDataKept=false --rtab-rss-cap-gb 6`,
seeds 0/1/2, slam2 = main@3d78f68 (sync, pre async-LC). Started 16:2x PDT 2026-10-22, stopped 22:08–22:12 PDT (~5.5 h wall, rt factor 1.15–1.26x).
Raw data: `~/slambench/res24h/seed{0,1,2}/` (metrics.jsonl, err_1hz.csv, frames/, snap/, rtabmap.db, slam2.pkl).

## How it was stopped / resume
- Each seed was stopped right after its 10-min metrics record (SIGINT; seeds 0/2 had SIGINT ignored, so SIGTERM 120 s later).
  The harness has no signal handler, so the final `slam.save`/`done.json` did not run. RTAB-Map got stdin EOF and ran `rt.close(true)`; all 3 dbs pass `PRAGMA quick_check`.
- Last slam2 graph = hourly save at t=21600 s (6.0 h) → `slam2.pkl` (copied to `slam2_final_t21600.pkl`), plus hourly `snap/`.
- **Resume is not supported**: life24 deletes rtabmap.db at start, never calls `Slam2.load`, and does not save sim/robot/RNG/chair state.
  Continuing would need a `--resume` (load slam2.pkl + reopen db + restore sim state). slam2 alone can be reloaded (`Slam2.load`, reload check passed every hour).

## Final (last record) — measured
| seed | sim h | path m | chair moves | slam2 ATE win / cum / max m | slam2 occ IoU | RTAB ATE win / cum / max m | RTAB IoU | odom ATE win / cum | GT-pose IoU |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6.67 | 4462 | 1 (at 6 h) | 4.12 / 2.72 / 4.70 | 0.160 | 4.14 / 2.86 / 16.9 | 0.079 | 5.47 / 2.76 | 0.491 |
| 1 | 6.17 | 3881 | 1 (at 6 h) | 1.46 / 0.42 / 2.46 | 0.212 | 6.08 / 5.77 / 18.8 | 0.082 | 3.81 / 6.22 | 0.493 |
| 2 | 6.33 | 4099 | 0 (6 h move placed nothing) | **0.14 / 0.19 / 1.23** | **0.350** | 3.57 / 1.80 / 8.3 | 0.148 | 5.10 / 2.77 | 0.496 |
| mean | | | | 1.91 / 1.11 / 2.79 | 0.241 | 4.60 / 3.48 / 14.7 | 0.103 | 4.79 / 3.91 | 0.493 |

slam2 graph/loops (final): nodes 588 / 348 / 308; loops accepted 116 / 66 / 61 of which **false 37 / 14 / 4**; long edges false 543/1257, 148/728, 95/711;
`loops_downweighted` (DCS) 18 / 11 / 5 (not split into true/false — the Phi=1 down-weighting of true loops is still present but unquantified here).
slam2 ms/update mean 8.7–9.2 (p95 20–25), flat over all hours; opt 16–27 ms; graph 4.4–6.9 MB, save 2.7–5.0 MB; harness RSS 1.36–1.76 GB (flat).
RTAB-Map: 15.3–17.7 k nodes (WM ~0.9–1 k), 1.6–9.8 k loops (40–271 false), 715–846 ms/frame, RSS ~3.0–3.4 GB, db 2.9–3.1 GB (~470 MB/sim-h), ~3.5–4.1 kCPU-s/sim-h. RSS cap (6 GB) never hit.

## Hourly trend (slam2 ATE window m / false loops / occ IoU) — measured
| h | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1 | 0.12 / 0 / 0.335 | 0.18 / 0 / 0.328 | 0.12 / 0 / 0.319 |
| 2 | 0.13 / 0 / 0.346 | 0.14 / 0 / 0.332 | 0.13 / 0 / 0.328 |
| 3 | **2.27 / 7** / 0.190 | 0.13 / 0 / 0.333 | 0.13 / 4 / 0.335 |
| 4 | 2.30 / 10 / 0.183 | 0.12 / 0 / 0.336 | 0.13 / 4 / 0.347 |
| 5 | 4.26 / 29 / 0.159 | 0.14 / 3 / 0.326 | 0.12 / 4 / 0.342 |
| 6 | 4.10 / 36 / 0.163 | **1.38 / 13** / 0.214 | 0.16 / 4 / 0.351 |
RTAB-Map ATE window by hour (mean of seeds): 1.29, 2.41, 3.42, 3.70, 3.72, 6.28 m — monotone degradation; db and RSS grow linearly.
Odom-only ATE window grows ~0.7–1 m/h (reference for "no correction").

## Map stability hour-over-hour (from hourly slam2 snapshots; re-render nodes present at h-1 with poses at h-1 vs h, 1-cell tolerant IoU)
seed0 0.97 0.90 0.99 0.95 0.98 · seed1 0.98 0.92 0.94 0.98 0.92 · seed2 0.98 0.98 1.00 1.00 1.00.
Old node poses essentially never move (median shift 0–1 mm, p95 ≤ 3 mm); IoU < 1 comes from scan refresh of existing nodes, not graph reshuffles.
Harness `stab_iou_1h` (GT-aligned accumulated grid) final: slam2 0.67 / 0.35 / 0.87, RTAB 0.15 / 0.21 / 0.45, GT 0.91–0.93.

## Verdict so far (6–6.7 sim-h, 3 seeds)
Measured:
- slam2 is cheap and flat: ~9 ms/update, ~1.5 GB RSS, ~0.8 MB/sim-h graph, no growth in cost over 6 h. RTAB-Map is ~90x slower per frame and grows ~470 MB db/sim-h.
- slam2 is accurate while it accepts no false loops (seed2 all 6 h, seed1 to 5 h, seed0 to 2 h: ATE ~0.12–0.16 m, IoU ~0.33–0.35 vs GT-pose ceiling 0.49).
- Failure mode = **false loop closures / relocalisations**: seed0 from h3 (7→37 false loops, ATE 2.3→4.1 m), seed1 at h5–6 around the first chair move (3→14 false, ATE 1.4 m).
  Once a false loop lands, the old map stays frozen and new nodes are built in a shifted frame (duplicate walls in seed0 final frame); no recovery observed.
- RTAB-Map degrades in all seeds (mean ATE win 1.3 → 6.3 m), so slam2 still beats it on every seed on ATE and IoU, but 2/3 seeds of slam2 failed within 6 h.
- Chair moves: only the 6 h move was reached (seed0 1 chair, seed1 1 chair, seed2 none placed). The 12/18/24 h moves were not reached.
Assumed / not measured:
- Behaviour beyond 6.7 sim-h, after more chair moves, and on the Orin (this is i777 RTX, sync slam2 without async-LC).
- True vs false split of down-weighted loops; whether async-LC changes loop acceptance (it should not change the math).
Next: tighten loop/long-edge acceptance (false-loop gate, e.g. require multi-hypothesis consistency or reject loops implying >X m jump vs odom), quantify DCS Phi=1 down-weighting of true loops, add `--resume` to life24.

Videos (h264, 10 fps, 1 frame / 2 sim-min, tiles GT | slam2 | RTAB | odom): `seedN_map_building_full_h264.mp4` + `seedN_final_frame.png` in `~/slambench/res24h/export/`.
