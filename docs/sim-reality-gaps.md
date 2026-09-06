# Sim ↔ reality gaps (Kevin / Anglerdroid)

Inventory of ways the lightweight 2D sim can diverge from the real Orin robot.
Priority: P0 = high impact on collision/stuck transfer, P1 = medium, P2 = polish.

## Dynamics / actuation
| ID | Gap | Reality | Sim today | Priority | Address |
|----|-----|---------|-----------|----------|---------|
| D1 | Instant v,w | ODrive + `VEL_RAMP_RATE=3` turns/s², ~1.6 m/s² | Instant set | P0 | 1st-order ramp / accel limits |
| D2 | Command latency | `LATENCY_S≈0.15` + 10 Hz twist_for | 0 | P0 | Delay queue of cmds |
| D3 | Diff-drive | `v_l/r = v ± ω·L/2`, L=0.34 m | Unicycle integrate | P0 | Wheel speeds + integrate |
| D4 | Speed caps | MPPI v_max=0.25, w_max=0.8; WB max_speed~50% | Policy cruise only | P0 | Clip to same limits |
| D5 | Slip / carpet | Real floor slip, caster scrub | Ideal rolling | P1 | Noise + slip factor |
| D6 | Battery / torque | Voltage sag, thermal | Infinite | P2 | Optional derate |
| D7 | Watchdog / stale | 5 s WD, command-stale idle | N/A | P1 | Sim stale-zero if cmd gap |

## Geometry / body
| ID | Gap | Reality | Sim today | Priority | Address |
|----|-----|---------|-----------|----------|---------|
| G1 | Mast / tall body | Mast hits table tops (`MAST_CLEAR_CM=45`) | Height map exists but collision is 2D FOOT only | P0 | 3D AABB / mast cylinder collide |
| G2 | Non-rect body | Wheels, casters, camera boom | Axis-aligned FOOT rect | P1 | Better footprint polygon |
| G3 | FOOT pads | Fwd/bwd/lat pads in config | Copied constants | OK | Keep import from robot_config |
| G4 | Moving furniture | Soft / pushed | Static | P2 | Later |

## Perception / maps
| ID | Gap | Reality | Sim today | Priority | Address |
|----|-----|---------|-----------|----------|---------|
| P1 | Depth noise / holes | RealSense noise, glare | Perfect occupancy | P0 | Dropout + noise model |
| P2 | Persistent map lag | Ego warp from VO drift | Perfect world→ego | P0 | Pose noise / drift |
| P3 | Partial FOV | Cameras don't see 360 | Full world known in ego crop | P1 | Mask unknown behind |
| P4 | Mast height sensing | From depth | Synthetic height | P1 | Box3D projects height |
| P5 | Dynamic people | Moving | Static | P2 | Agents later |

## Control / software loop
| ID | Gap | Reality | Sim today | Priority | Address |
|----|-----|---------|-----------|----------|---------|
| C1 | Async rates | Vision 30 Hz, house_bot 1 Hz, twist_for 10 Hz | Single tick | P0 | Multi-rate sim |
| C2 | Safety vs planner race | Threads, clear() mid-tick | Sync-threaded | P1 | Inject races in tests |
| C3 | MPPI not in sim loop | Live uses NumPy MPPI | HouseBotLite / GoalSeek | P0 | Wire sim MPPI backend |
| C4 | Speech / faces | Side channel | Separate package | P2 | Optional |

## Environment
| ID | Gap | Reality | Sim today | Priority | Address |
|----|-----|---------|-----------|----------|---------|
| E1 | Real house layout | Messy rooms | Hand rectangles | P1 | Trace from atlas / Box3D room |
| E2 | Soft obstacles | Couch cushions compress | Hard binary | P1 | Soft collision margin |
| E3 | Doorways / thresholds | Lip, carpet edge | Flat | P2 | |

## Strategy
1. Keep fast 2D as unit-test harness.
2. Upgrade 2D robot model (D1–D4) immediately.
3. Add **Box3D** layer: furniture as axis-aligned 3D boxes → project to occ+height; mast collision = 3D.
4. Later: optional PyBullet/MuJoCo if Box3D isn't enough — not required to start.
5. Never re-arm live until P0 gaps have tests or known residuals documented.
