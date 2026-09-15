# sim/rl_nav PLAN — bag-matched ego + unicycle PPO

Living doc for the from-scratch 5 Hz PPO that must **look and operate like
Kevin**. Do not load insect / Newton ckpts. Do not touch `sim/explore/`.

Wrote 2026-09-13 after piloted bags on Orin. Update here, not only in chat.

## Decision (obs representation)

**Avenue 1 only:** RealSense → early decimate → ego UNKNOWN / CLEAR / OBSTACLE,
same tensor in sim and on Kevin.

- Not unprojected 640×480 in the net (avenue 2).
- Not a new preprocess stack (avenue 3) unless a bag rebuild proves GSD / fill
  / `k_obs` cannot match on the same scene.
- Raw z16 stays on disk so we can relabel. Policy still sees 80×60 labels.

James 2026-09-13: as a human, **height_cm** is a better nav view than a 1-bit
obstacle map (low bumps that are not “obstacles” still matter). Keep 1-bit as
the drive/crash contract for v1; do not block match+train on a height channel.
If we add it later it is an extra ego plane from the same scatter, not a new
stack. Soft-low already uses height on live. Sim of continuous height is
harder than UNKNOWN/CLEAR/OBSTACLE.

Live contract: 320×240 @ 1 cm (`RCX=81`, `RCY=119`, +X forward / right on
image). Policy train spec is 4× that: **80×60 @ 4 cm** (`EGO80_*` in
`src/perception/fast_ego80.py`). Sim: `sim/rl_nav/kernels.py` `EGO_H=60`
`EGO_W=80`. Old 40×48 `ckpt.pt` will not load — park it.

## Bags (source of truth)

On Kevin: `/home/jetbot/.kevin/bags/<name>/`.
On i777: `sim/rl_nav/bags/<name>/` (rsync; z16 last).

Each bag (slim motion-only unless noted):

| file | contents |
|------|----------|
| `rs1_z16.bin` / `rs2_z16.bin` | uint16 depth, row-major H×W per frame |
| `imu.npy` | D435i IMU |
| `wheels.npy` | encoder / wheel vel |
| `cmds.jsonl` | every `/twist` (also mirrored in `sim/rl_nav/out/pilot_cmds.jsonl`) |
| `stamps.npy` + `meta.json` | time + **actual** depth size + fx/fy/ppx/ppy |

Keep: `wander_5min_pilot`, `wander_pilot_d`, `wander_pilot_f`, `wander_pilot_g`.
`wander_pilot_c` has imu/wheels/cmds/labels; its z16 was deleted on Kevin to
free disk (sit-heavy). Skip empty `wander_pilot_e`.

**USB2 capture for these bags is 640×480**, not 848×480. `meta.json` `"rs"` and
`intrinsics.*.width/height` are authoritative. Example RS1:
`fx=fy≈380`, `ppx≈316`, `ppy≈233`, `width=640`, `height=480`.

## 848 vs atlas: how to crop / align (do not reinvent)

Native D435 depth default is **848×480**. The atlas/ego canvas is **320×240**.
There is **no center crop of 104 px per side** on the depth image.

The crop that keeps the valuable ground is the **ego blit after RS1’s 180°
flip**, in `src/vision.py`:

1. Scatter RS1 verts into a **camera-centered** 320×240 @ 1 cm (same canvas
   whether the stream was 848 or 640; width only changes GSD/FOV).
2. Flip `obs1 = z1[::-1, ::-1]` (RS1 is mounted inverted; atlas color is
   `rs1.color[::-1, ::-1]`).
3. Horizontal shift **`TD_X_OFFSET = -75`**. After the shift, the RS1 known
   rectangle is clipped to `td_col_end = 320 + (-75) = 245` — keep **forward
   of the axle**, drop ~75 cm behind camera center.
4. Axle stays at **`(RCX, RCY) = (81, 119)`**.
5. RS2 is locked to that: `fw_x = TD_X_OFFSET + FW_TD_X_DELTA` (`-75 + 132 =
   +57`), then `rot90` CW.

80×60 uses the same geometry: **`TD_X_OFFSET80 = -19`** (`≈ -75/4`) in
`fast_ego80.py`. Unstuck is a second axle-relative window (0.80 m back /
2.40 m forward / 0.96 m lateral) — same “valuable = in front of axle”
idea, not optical center.

**Rebuild rule:** deproject with bag `width/height` + `ppx/fx`, then **flip +
`TD_X_OFFSET`**. Extra 848 columns, when present, fall off the 320 canvas
after the −75 blit — that *is* the crop. `from_bag.py` still hardcodes
`480×848`; that is a bug for these bags.

## Operate-like (kinematics from the same bags)

5 s `/twist` batches, clamp fwd ±0.25 m/s, ang ±0.8 rad/s. Sim `clip_vw`
adds a **small** momentum haircut: at vmax, 85% of wmax remains (40 lb,
not a hard wheel diamond that left zero yaw at speed). Slow down to get
full steer, or crash. Coverage paint is the whole objective (go
everywhere, as fast as the cap). No goal / hint bearing — goal-following
is a later behavior. Feel on carpet (safety 1): `0.10 m/s × 5 s` →
~0.35–0.44 m; `0.20–0.25 rad/s × 5 s` overshoots math (wrap `dyaw` in
logs). Pinched `fwd_scale` → 0–9 cm; `bwd=0` reverse is a no-op. Fit
unicycle / wheel+IMU to `cmds.jsonl` + `wheels.npy` + pose0/pose1 before
trusting sim `k_obs`.

## Work order

1. Finish rsync of `f`/`g` z16 onto i777.
2. Fix `from_bag.py` rebuild for meta size (640×480) + flip + `TD_X_OFFSET80`. **Done** (`bag_io.py` + `ego80.py` copy).
3. Rebuild 80×60 from z16; compare to `labels80_live.npy` where it exists. **Probed** `wander_5min_pilot` + `d`. Live 80×60 is sparse (downsample of 320 when labels skipped) — **rebuild from z16 is the train spec**, not `labels80_live`.
4. Match sim `k_obs` **look** to bag rebuild (FOV silhouette, 3-tone). **In progress** — baked stencil, no deproject in the 5 Hz loop. Kevin clear/unknown model can move later to match whatever the sim settles on.
5. Fit 5 Hz unicycle to logged twists; then train from scratch. Park old `ckpt.pt`.

## Look-alike contract (2026-09-13)

Policy transfer is **pixels of the labeled 80×60**, not cameras and not projection math.

Bag rebuild tiles are a **green RS1 house-footprint + red/green RS2 cone + black outside + blue SELF**. The old sim AABB painted ~75% CLEAR and looked like a filled rectangle. That will not transfer.

**Sim does not deproject z16.** It paints world occupancy into a bag-baked FOV stencil (`sim/rl_nav/fov80_rs1.npy`, from ~500 rebuild frames). `k_obs`:

- Outside stencil → UNKNOWN
- Inside stencil → CLEAR, or OBS if `world_occ`, ~9% holes
- RS2 first-hit band → OBS; behind hit and past `RS1_X1` → UNKNOWN
- SELF is 0 in the policy tensor (same as UNKNOWN); HUD overlays blue

HUD LUT in `dash.py` uses the same dark / green / red / blue as bag contact tiles.

`labels80_live` agree ~0.55 with rebuild because live is mostly UNKNOWN. Ignore it for `k_obs` fit.

## Probe 2026-09-13 (i777)

`sim/rl_nav/probe_bag.py` → `out/bag_probe/contact_*.png` (depth | rebuild | live) and `look_*.png` (rebuild | sim).

| source | unk | clear | obs | self |
|--------|-----|-------|-----|------|
| bag 5min rebuild (n=410, **pre-pitch**) | 0.39 | 0.48 | 0.11 | 0.015 |
| bag 5min rebuild (n=410, **pitched RS2**) | 0.41 | 0.52 | 0.056 | 0.015 |
| bag d rebuild (n=430) | 0.39 | 0.45 | 0.15 | 0.015 |
| sim `k_obs` (N=64 spawn, **pre-stencil**) | 0.18 | 0.75 | 0.06 | (baked into 0) |
| sim `k_obs` (N=64, pitched stencil + fat) | 0.40 | 0.53 | 0.07 | (baked into 0) |

`sim/rl_nav/compare_transfer.py` writes `out/transfer/rs_like_sim.png` (real RS z16 vs sim ego).

**Look knobs are runtime** (`look[0..10]` on `k_obs`, `env.set_look`). Changing paint constants no longer recompiles Warp (~2.5 min).

## Transfer look (2026-09-14) — Kevin live = sim `k_obs`

Policy transfer is the **80×60 float** (`UNKNOWN/SELF=0`, `CLEAR=0.5`, `OBS=1`,
hull throttle `0.20`). Same HUD LUT as `dash.py` (dark / green / red, blue
SELF after nearest upscale). 320×240 atlas/reflexes stay on capture; this
tensor is extras ~3 Hz via `label_from_z16`.

| Step | Change | Verify |
|------|--------|--------|
| 1 | One hull on Kevin `fast_ego80` = `in_self_body` / sim `ego80` | `compare_transfer.py` `step1_pass` (pixel agree >0.999 on same z16) |
| 2 | Kevin extras writes `~/.kevin/policy_ego80.png` with that HUD | Eye: house∪cone, one blue rect, black outside — same as sim ego panel |
| 3 | Bag `labels80_live` (320 downsample) vs z16 rebuild | `old_live_vs_kevin_z16_agree` — live was sparse; extras path is the feed |
| 4 | Tone rates rebuild vs sim `k_obs` | unk/clear/obs within ~5pp; `rs_like_sim.png` |
| 5 | Throttle sectors from `fwd/bwd/ang_scale` | Amber on hull when pinched; sim HUD match |
| 6 | On-bot vs on-sim stills | Kevin png vs watch ego panel / `out/transfer/rs_like_sim.png` |

Do not match sim to Kevin. Kevin moves to the sim tensor.

James 2026-09-13 evening: do **not** start overnight until the look is approved. Obstacles are **static top-down** footprints (`world_occ` + unknown behind first hit). Floor in FOV/cone is clear — no 1 px dirt. Domain rand is **rare ephemeral blobs** (~24 cm, ~1.4 s life, `BLOB_P=0.006`). Same hash in paint + `foot_ok` + throttle rays — they are real obstacles, not a tell to ignore. Show/hide across frames is intended; the policy should slow down and use the 64-byte memory. No per-frame flicker carpet. Coverage objective unchanged.

SELF at 80×60 is **one hull** (36×44 cm) covering body + 10 cm wheels + mast — strips tire returns without CAD wheel ears. Live 1 cm `fast_ego80` still uses the four `robot_config` boxes.

Throttle source is **on the ego pixels** the policy sees: front / back / left-wheel / right-wheel of the hull go to `THROT_MARK=0.20` when that side’s clearance is inside the live-ish pinch band (`THROT_M=0.28` fwd, 0.18 side, 0.20 back). HUD stamps those sectors amber. Same scales pinch `+v` / `−v` / `+w` / `−w`. On Kevin, stamp the same sectors from `fwd_scale` / `bwd_scale` / `ang_scale` + nearest obs side.

Remaining tell was partly a **buggy bag rebuild**: RS2 phys_h used `25.6-90` (=64.4° down). Floor in the wedge looked like OBS (side-view slope ~20°). Fit 2026-09-14: **27.0° down from horizontal**, `cam_h=0.475`. BEV `FW_ROTATION` still uses 25.6-90.

Blind 2026-09-13: first stencil pass easy (speckle + thin red rim). After thicker hits + unknown-behind-cone, still easy — occlusion keep-prob became a **green dither fog** at the far cone (16/16). Solid unknown behind hit, clustered holes. Close-hit pie-fill matched rates (40/46/13 vs 39/48/11) but red was a CAD wedge. Next: blob depth not pie-to-2.5 m, two-scale holes, wiggly cone / hit.

## Live on Kevin (2026-09-14)

First on-bot run of this PPO: `--local-planner neural_rl --wander` (no
`--house-bot`). Orin has no torch — export `ckpt.pt` to ONNX and drop it at
`~/.kevin/rl_nav.onnx`. `src/rl_nav_live.py` loads that, 5 Hz hold, 64 prims,
64-byte mem. Safety scales the twist once in `wheelbase`; if the nose is
pinned (`fwd_scale < 0.20`) and the net is not turning, inject 0.45 rad/s
yaw so the 5 s zero-vel watcher does not idle the ODrives (blue).

Bring-up that actually drove:

```
CAN_IFACE=can1 RS_DEPTH_W=640 RS_DEPTH_H=480 RS_FPS=30 \
KEVIN_RL_NAV_CKPT=~/.kevin/rl_nav.onnx \
python -u main.py --slam self --wheel-imu-prior --no-rerun \
  --auto-local --local-planner neural_rl --wander \
  --rs1 815412070676 --rs2 944622074292
```

Do not commit bags, `out/ckpt.pt`, or overnight dumps.

## Related

- `src/vision.py` — `TD_X_OFFSET`, `_build_obs_mask`, RS1 180° flip
- `src/robot_config.py` — `RCX/RCY`, `EGO_PX_SIZE`
- `src/perception/fast_ego80.py` — 80×60 + `TD_X_OFFSET80`
- `docs/perception/CONTRACT.md` — label semantics
- `docs/kevin-autonomy-midlayer.md` — 30 Hz atlas / ego
- `AGENTS.md` — Orin 30 Hz budget, decimate metric
