# tools/realsim — photoreal casita for Kevin (M1)

NVIDIA **ovrtx** (Omniverse RTX, pre-release Python API, *Real-Time Path-Tracing*) renders the ONE world model
(`src/world_params.py`: walls STL + counters + `floor_zones`) dressed by `dressing.json`, from the real camera poses and
intrinsics, then `camera_model.py` turns the linear HDR into what the robot's cameras deliver (vignette, MTF blur, AE,
AWB excluding clipped pixels + measured residual cast, shot/read noise, JPEG).  Frames, textures and furniture are NOT
in git (CC0 assets + `LICENSES_manifest.json` live in `~/kevin_bench/m1/assets`).

| file | role |
|---|---|
| `poses.py` | twin camera mounts, model-edge vs real LSD-segment chamfer score, overlays |
| `local_pose.py`, `final_poses.py` | joint position fit per in-place-turn group, per-look heading, `poses.json`, lamp xy from saturated blobs |
| `usd_scene.py` | USDA builder (walls/damask/ceiling/battens, floor zones, counters, blockout boxes, drapes, CC0 furniture refs, lamps + fixture diffusers, cameras with OpenCV lens distortion, RenderProducts) |
| `dressing.json` | look: materials (MEASURED vs ASSUMED in `_src`), lights, furniture, cameras, carpet ridges |
| `carpet_ridges.py` | bakes lengthwise carpet ridges into the carpet normal map (hard-floor zones cover them) |
| `ovrtx_render.py` | `RtxSim`: open stage via ovstage, move cameras (`omni:xform`), step, map HdrColor/LdrColor |
| `render_looks.py`, `montage.py`, `colour_match.py`, `lpips_eval.py` | reference renders, real/sim montage, per-surface colour match, masked LPIPS |
| `bench_fps.py` | fps per configuration (moving camera, LdrColor read back to CPU each frame) |
| `rtx_bridge.py` | live MuJoCo -> ovrtx: follows `/dev/shm/kevin_sim.json`, writes `/dev/shm/kevin_rtx_<cam>.jpg` |

Run (i777): `XDG_CACHE_HOME=/dev/shm/ovcache` (shader cache, ~0.9 GB, lost on reboot: first start compiles ~11 min,
new material permutations a few minutes; warm start ~5 s).  Venv `~/.venv-ovrtx` (ovrtx 0.6.0, ovstage 0.3.0).
30 fps setting: `omni:rtx:post:dlss:execMode = "performance"` (DLSS-SR; DLSS-RR/NR and frame generation are not
available on the 3090 / driver 595.99).
