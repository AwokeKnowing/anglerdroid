# Host Simulator Quickstart (i777)

High-fidelity simulator for Kevin's autonomy stack. Runs on i777 workstation with RTX 3090 for offline transfer learning and policy iteration.

## Setup (i777)

```bash
cd /workspace

# Install dependencies (if not already installed)
pip install numpy torch imageio opencv-python

# Optional: Rerun for visualization
pip install rerun-sdk
```

## Quick Start

### Basic Scenarios

```bash
# Table overhang test (30 seconds, high fidelity)
python -m sim.host_sim --scenario table_overhang --duration 30 --fidelity high

# Soft bed scoring
python -m sim.host_sim --scenario soft_bed --duration 60 --fidelity medium

# Tight doorway navigation
python -m sim.host_sim --scenario tight_doorway --duration 40 --fidelity medium

# Combined house (all features)
python -m sim.host_sim --scenario combined_house --duration 120 --fidelity low --seed 42
```

### Generate Demo GIFs

```bash
# Auto-saves to artifacts/sim/ if --save omitted for short runs
python -m sim.host_sim --scenario table_overhang --duration 20 --fidelity high

# Explicit save path
python -m sim.host_sim --scenario soft_bed --duration 30 --save ~/bed_demo.gif
```

### Rerun Visualization

```bash
# Enable Rerun logging
python -m sim.host_sim --scenario hallway_pinch --duration 60 --rerun --save /tmp/sim.rrd

# View in Rerun
rerun /tmp/sim.rrd
```

## Available Scenarios

### Enhanced (High-Fidelity)

- **`table_overhang`**: Dining table with mast crash case. Tests topdown near-field reflex (<30cm).
- **`table_overhang_tight`**: Table with narrow side clearances (must navigate around).
- **`soft_bed`**: Dog bed with reduced MPPI cost (0.5× normal obstacle).
- **`keepout_mat`**: Checkered keepout zone (visual boundary, no physical obstacle).
- **`tight_doorway`**: 70cm doorway requiring precise alignment.
- **`hallway_pinch`**: Long corridor with 60cm pinch point.
- **`combined_house`**: Full house with table, bed, keepout, and doorway.

### Base (from `sim/world.py`)

- **`empty`**: Clear space (smoke test).
- **`couch_pinch`**: Couch ahead, must reverse or spin.
- **`house`**: Full house with walls, couch, table, chairs.
- **`hallway`**: Corridor with dead-end.
- **`doorway`**: 80cm doorway (wider than tight_doorway).
- **`cul_de_sac`**: U-shaped pocket ahead.

## Fidelity Modes

### `--fidelity none`
- No sensor noise, no latency, no drift
- Instant dynamics (unicycle model)
- ~100× real-time on i777
- **Use for**: Quick smoke tests, algorithm debugging

### `--fidelity low` (Default for fast iteration)
- No sensor noise
- Differential-drive dynamics with accel limits
- No latency
- ~50× real-time on i777
- **Use for**: Policy development, parameter tuning

### `--fidelity medium` (Recommended for transfer validation)
- Depth dropout: 0.8%, noise σ=3mm @ 1m
- Pose drift: 2mm/step
- Command latency: 50ms
- ~10× real-time on i777
- **Use for**: Transfer testing before deploying to Kevin

### `--fidelity high` (Close to hardware)
- Depth dropout: 1.5%, noise σ=8mm @ 1m
- Pose drift: 4mm/step
- Command latency: 150ms
- Sensor jitter: ±5ms
- ~1-3× real-time on i777
- **Use for**: Final validation, stress testing

## Policy Selection

```bash
# MPPI (default, matches live Kevin)
python -m sim.host_sim --scenario house --policy mppi

# House_bot FSM (state machine)
python -m sim.host_sim --scenario couch_pinch --policy housebot

# Goal-seeking planner
python -m sim.host_sim --scenario doorway --policy goalseek

# Random walk (baseline)
python -m sim.host_sim --scenario empty --policy random

# Stop (test initial conditions)
python -m sim.host_sim --scenario table_overhang --policy stop --duration 5
```

## MPPI Soft Cost

```bash
# Soft prefer costs (default ON)
python -m sim.host_sim --scenario hallway_pinch --soft-cost

# Hard mask only (legacy)
python -m sim.host_sim --scenario hallway_pinch --no-soft-cost
```

Soft costs make MPPI prefer clearer paths even when hard safety allows motion. See `docs/sim-reality-gaps.md` for transfer metrics.

## Reproducibility

```bash
# Fixed random seed for deterministic runs
python -m sim.host_sim --scenario house --seed 42 --duration 120

# Sweep seeds for robustness testing
for seed in 11 42 99 123; do
    python -m sim.host_sim --scenario tight_doorway --seed $seed --duration 40 --save door_$seed.gif
done
```

## Running Tests

```bash
# Host scenario tests (table overhang, soft bed, etc.)
pytest sim/test_host_scenarios.py -v

# Sensor noise tests
pytest sim/test_host_scenarios.py::TestSensorNoise -v

# All sim tests
pytest sim/ -v
```

## Example Output

```
Running table_overhang for 30.0s (900 steps @ 30.0Hz)
Fidelity: high, Policy: MppiSimPolicy
  Step 30/900: pos=(0.81, 1.23) θ=2.3° v=0.15 w=0.05 safety=(f:0.98 b:1.00 a:1.00) RT×2.1
  Step 60/900: pos=(0.86, 1.24) θ=1.8° v=0.12 w=-0.02 safety=(f:0.87 b:1.00 a:1.00) RT×2.3
  ...
  Step 570/900: pos=(1.42, 1.19) θ=-0.5° v=0.00 w=0.00 safety=(f:0.00 b:0.85 a:0.45) RT×2.5

============================================================
SIMULATION COMPLETE
============================================================
Scenario: table_overhang
Fidelity: high
Steps: 900 (30.0s sim time)
Wall time: 12.3s (×2.4 real-time)
Final pose: (1.42, 1.19) θ=-0.5°
Collisions: 0

✅ PASSED: No collisions
```

## Integration with Live Stack

Simulator imports live modules directly:

```python
from src.safety import SafetyGuard
from src.mppi_costmap import MppiCostmapPlanner
from src.robot_config import RCX, RCY, EGO_PX_SIZE
```

**Same data formats**: obs_map (320×240 uint8), safety scales (fwd/bwd/ang), twist commands (fwd_mps, ang_rads).

**No conversions needed**: Validated policies can be deployed to Kevin without modifications.

## Performance Benchmarks (i777 + RTX 3090)

| Fidelity | Real-Time Factor | MPPI Time/Tick | Wall Time (30s sim) |
|----------|------------------|----------------|---------------------|
| None     | ~200×            | N/A (stub)     | 0.15s               |
| Low      | ~50×             | <1ms           | 0.6s                |
| Medium   | ~10×             | ~2ms           | 3.0s                |
| High     | ~2-3×            | ~4ms           | 10-15s              |

**Bottleneck**: MPPI sampling (512 trajectories × 30 steps) on GPU. PyTorch `grid_sample` gives ~2-4ms on RTX 3090.

## Known Limitations (v0)

See `docs/host-simulator-architecture.md` "Remaining Fidelity Gaps" section:

1. **RGB synthetic**: Blank frames; house_bot face/keepout triggers stubbed.
2. **Visual odometry**: Simplified correction model (no dense optical flow).
3. **Soft obstacle physics**: MPPI cost model only; no actual deformation.
4. **Dynamic obstacles**: Static world; no moving people/pets.
5. **RealSense IR patterns**: Statistical depth noise, not ray-traced speckle.

**Mitigation**: These gaps are acceptable for policy transfer. Validate critical scenarios on hardware before live deployment.

## Troubleshooting

### "Module not found: rerun"
```bash
pip install rerun-sdk
```

### "Module not found: imageio"
```bash
pip install imageio
```

### GIF too large / playback slow
- Reduce `--duration` (GIFs are sampled every 3rd frame)
- Use `--fidelity low` for faster runs
- Save as Rerun RRD instead: `--rerun`

### Simulation slower than expected
- Check GPU usage: `nvidia-smi` (should show PyTorch process)
- Reduce MPPI samples: Edit `sim/mppi_policy.py` (n_samples=512 → 256)
- Use `--fidelity low` (skips sensor noise computation)

### Robot gets stuck / oscillates
- Known issue: MPPI can get stuck in local minima (cul-de-sac scenarios)
- Try different seed: `--seed 42`
- Adjust MPPI params: Edit `src/mppi_costmap.py` (temperature, horizon)

## Next Steps

1. **Tune policies**: Run seed sweeps, analyze failure cases
2. **Validate transfer**: Compare `--fidelity medium` vs live hardware metrics
3. **Stress test**: Combined house, long duration (5+ minutes)
4. **Playback mode** (future): Record atlas from live runs → replay with different policies

## Contact / Issues

Branch: `cursor/host-simulator-8d47`  
Docs: `docs/host-simulator-architecture.md`  
Tests: `sim/test_host_scenarios.py`

For questions, see repo README or `docs/kevin-autonomy-midlayer.md`.
