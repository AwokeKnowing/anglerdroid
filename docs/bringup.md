# Kevin real-robot bring-up (Orin "jetbot", code in ~/anglerdroid on branch fullstack-sim-slam-2026-10-14, venv ~/.venv-kevinnav)

Emulator rehearsal of this whole flow (i777): `~/.venv-kevinsim/bin/python emu/tools/rehearse_bringup.py <tag> <pitch_deg>`.

Order matters; stop at the first FAIL. Physical e-stop (mast) within reach the whole time.

0. **USB 3 for both RealSenses (blocker found 2026-10-14):** both cameras currently enumerate at USB 2.1 (480M hub
   chain with the wifi dongle + CANable) -> 848x480 depth fails, D435i IMU "not available", RS1 delivered no frames.
   Plug the D435 + D435i into the Orin's 10 Gb/s port (or a USB 3 hub on it), short USB-C 3.x cables.
   `lsusb -t` must show them under `Bus 002 ... 10000M`.
1. **Pre-flight (read-only, never moves motors):** `cd ~/anglerdroid && ~/.venv-kevinnav/bin/python scripts/bringup_check.py`
   — roles by product name, USB 3, RS1/RS2 30 Hz, IMU gravity + stamps, ODrive CAN heartbeats (both axes, no errors),
   safety envelope unit checks, calibration.json sanity. All PASS before going on.
2. **Wheels up** (chassis on a box, both wheels free): `scripts/bringup_run.sh 0.3`, then
   `drive 0.1 0 2` / `drive -0.1 0 2` / `drive 0 0.5 2` / `drive 0 -0.5 2` via the page http://jetbot:8080 or `curl -s localhost:8080/api/command -d '{"text":"drive 0.1 0 2"}'`.  Check: correct wheel
   directions, forward = +x, left turn = CCW, wheel speed in the log matches command, stops on `stop`.
3. **E-stop:** while `drive 0.1 0 5` runs wheels-up, hit the e-stop -> wheels stop within ~0.2 s, nav logs the
   drive fault / CAN state; release + `stop`, confirm it does not resume on its own.
4. **Hand over the lens:** wheels-up, `drive 0.2 0 5`, cover RS1 (top-down) -> stops (`no_depth`); uncover, cover
   RS2 only -> RS1 still drives safety (expected: keeps going unless RS1 sees something). Desk/too-close = obstacle.
5. **Floor, slow, capped:** robot on the floor, clear 3 m ahead, `scripts/bringup_run.sh 0.3` (hard cap on every
   command, reverse <= 0.2, w <= 1.0).  `drive 0.2 0 3`, then a goal 2 m ahead.  Walk a box in front -> it stops.
   Park 10 s before the first drive: auto-cal logs `autocal: roll .. pitch ..` and writes src/calibration.json only if
   within 5 deg / 3 cm of the twin (warns if >1 deg).
6. **Replay log:** every run records to ~/kevin_logs/<stamp>/ (t*.npz per tick: RS1/RS2 depth, IMU, wheels, command)
   + nav.log. Copy to i777 for replay/sim comparison.
7. **Ruler tests (calibrate reality to the twin):**
   - straight: `drive 0.2 0 5` on a taped 1 m line; measured distance / odom distance -> `wheel_r_scale`
   - spin: 4 x 360 deg in place (`drive 0 0.5 50.3`), measured heading error -> `track_scale`
   - camera: box at 1.00 m (tape) in front; RS2 range vs tape; floor flat in the map (auto-cal)
   Put scales in src/calibration.json (`wheel_r_scale`, `track_scale`); twin stays nominal.
