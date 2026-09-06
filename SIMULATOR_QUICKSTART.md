# Simulator Quick Start

## ⚠️ LIVE ROBOT STATUS

**LIVE DRIVING IS STOPPED** following couch crash incident.

Use the `sim/` package for offline policy development.  
Use the `faces/` package for offline face recognition practice.

## Simulator Quick Commands

### Smoke Test (Required)
```bash
python -m sim.run --steps 200 --scenario empty
```
Expected: ✅ PASSED: No collisions

### Test Suite
```bash
python -m sim.test
```
Expected: ✅ All 7 tests passed

### Couch Pinch Scenario
```bash
python -m sim.run --steps 300 --scenario couch_pinch --policy housebot
```
Tests recover behavior when facing obstacle.

### Full House Navigation
```bash
python -m sim.run --steps 500 --scenario house --policy housebot --save house.gif
```
Generates visualization (requires `imageio`).

### Crash Hypothesis Test
```bash
python test_crash_hypothesis.py
```
Demonstrates that simulator enforces safety hard stops.

## Documentation

Full guide: `sim/README.md`  
Face recognition: `faces/README.md`

## Face Recognition (New!)

Offline-safe face enrollment and conversational greetings:

```bash
# Enroll faces
python -m faces.cli enroll "John Doe" photo.jpg

# List enrolled people
python -m faces.cli list

# Recognize from image
python -m faces.cli recognize photo.jpg

# Live demo (requires opencv-python)
python -m faces.demo webcam --tts stub --volume 0.1
```

See `faces/README.md` for installation and usage.

## Safety Note

The simulator enforces that policies **CANNOT** override SafetyGuard when:
- `fwd_scale=0` → forward motion BLOCKED
- `bwd_scale=0` → backward motion BLOCKED
- `ang_scale=0` → angular motion BLOCKED

This prevents the couch crash scenario where recover/commit forced motion through obstacles.

## Next Steps

1. Run smoke test to verify installation
2. Develop policy in simulator
3. Validate zero collisions
4. (Optional) Set up face recognition for conversational greetings
5. Port to robot ONLY after sim validation
