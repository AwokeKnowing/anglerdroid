# Live RS1 30 Hz check, 2026-10-17 (all times PDT)

The change in f9e2036: emu.world renders RS1 depth (and RS2 depth) first on every physics tick in a fast lane that has its own renderer. Colour, IR, mast and viewer frames run in a separate slow lane on their own schedule. The slow lane only drops off pending colour/IR fields, so it can never hold up a depth frame. Also merged: PR #62 commit 2203aca (wall collision skin, KEVIN_SAFETY_LOG). Safety logic, reaction time, staleness check and brake ramp are unchanged.

Restarts: emu.world at 16:23:45; nav and viewer at 16:24:35 (via dbg/restart_nav.sh, because nav does not reattach to recreated shm). cloudflared was not restarted. Kevin is idle.
Load during measurement: sim + nav + viewer + LightNav (vLLM) + ollama. GPU cap 300 W, graphics clock locked at 1695 MHz.

| metric | before (16:2x, old world) | after (16:25:34, 60 s) |
|---|---|---|
| kevin_rs1 shm rate | 18.47 Hz, interval p50 49.3 ms, p95 79.4 ms | **29.97 Hz, interval p50 33.4 ms, p95 38.6 ms, max 62.1 ms** |
| kevin_rs1 shm age when read | p50 58.0 ms, p95 73.6 ms | p50 11.8 ms, p95 17.6 ms |
| kevin_rs2 shm rate | 18.47 Hz | 29.97 Hz (interval p50 33.5 ms, p95 39.9 ms) |
| mast | 8.92 Hz | 8.94 Hz |
| colour lane (world log) | ~19 Hz (shared with depth) | 16.6-19.0 Hz |
| frame age at safety decision (safelog: now - t_frame) | p50 119.3 ms, p95 169.1 ms, max 199.3 ms | **p50 84.6 ms, p95 104.8 ms, max 140.2 ms** |
| GPU power | 126 W | p50 129.1 W, p95 130.8 W, max 131.7 W of 300 W; 16.5 GB used |

What is left of the after-restart latency is in nav, not the sim:
- nav ticks every ~56 ms (about 17.8 Hz), so it uses about every other 30 Hz frame.
- Frame age at tick start (t0 - t_frame) is p50 59.9 ms, p95 77.2 ms, against 11.8 ms age in shm. The roughly 48 ms gap is HAL/nav frame pickup.
- Tick compute (now - t0) is p50 23.7 ms, p95 29.9 ms.
These are nav/HAL files and were left untouched.
