# Simulation Curriculum (OMNI-EPIC-Inspired)

**Status**: Stub for future implementation.

## Overview

OMNI-EPIC-inspired curriculum generation for Kevin neural RL training:
- Foundation model generates scenario code + rewards
- Open-ended curriculum adapts to agent skill level
- Train in host-sim (i777), deploy ONNX to robot
- Complements ASPIRE: curriculum (pre-training) + skill refinement (post-deployment)

## Architecture (Planned)

```
FM (Gemini/Claude/GPT)
    ↓
[Scenario Generator]  ← current agent skill level, training history
    ↓
scenarios/<scenario_name>.py  ← procedural env code
rewards/<scenario_name>_reward.py  ← task-specific rewards
    ↓
[RL Training Loop]  ← PPO/SAC in host-sim
    ↓
models/policy.onnx  → deploy to robot (neural_rl.py)
    ↓
[Real-world execution]  → trace_logger.py
    ↓
[ASPIRE skill refinement]  ← failures → skill library updates
```

## Curriculum Scenarios

**Initial set** (hand-coded, then FM-generated variants):

### 1. Doorway Pinch
Narrow passages requiring precise turning:
- Width: 40-80cm (robot width = 42cm)
- Success: pass through without collision
- Failure: collision with doorframe
- Reward: goal reached - collision penalty - time penalty

### 2. Dog Bed Soft
Deformable obstacles (cushions, soft furniture):
- Height: 5-30cm, compliance: high
- Success: navigate around or slow approach
- Failure: aggressive contact (high velocity)
- Reward: goal reached + gentle contact bonus - aggressive penalty

### 3. Overhang Approach
Table undersides, shelves (mast clearance):
- Clearance: 30-70cm (mast height = 45cm threshold)
- Success: detect and avoid or navigate under if safe
- Failure: mast collision with overhead structure
- Reward: goal reached - overhead collision penalty

### 4. Checkered Keepout
Visual pattern-based forbidden zones:
- Pattern: high-contrast checkered, various scales
- Success: detect and avoid entry
- Failure: enter keepout zone
- Reward: goal reached - keepout entry penalty

### 5. Person Approach
Social distance maintenance (dynamic agents):
- Distance: 1.2-1.5m target (comfortable social space)
- Success: reach conversation distance without encroachment
- Failure: too close (<1.0m) or startling person
- Reward: social distance optimality - encroachment penalty

## Generator Interface (Stub)

```python
class CurriculumGenerator:
    """FM-based scenario code generator."""
    
    def __init__(self, fm_client, base_scenarios_path="sim_curriculum/scenarios"):
        self.fm = fm_client
        self.base_path = base_scenarios_path
    
    def generate_scenario(self, scenario_type: str, difficulty: float, agent_skill: dict) -> str:
        """Generate scenario code adapted to agent skill level.
        
        Args:
            scenario_type: Base scenario ("doorway_pinch", "dog_bed_soft", etc.)
            difficulty: Target difficulty 0.0-1.0 (learnable but interesting)
            agent_skill: Current agent capabilities (success rate, failure modes)
        
        Returns:
            Path to generated scenario code
        """
        raise NotImplementedError("FM curriculum generator not implemented")
    
    def generate_reward(self, scenario_path: str, success_criteria: dict) -> str:
        """Generate reward function for scenario.
        
        Args:
            scenario_path: Path to scenario code
            success_criteria: Task-specific success conditions
        
        Returns:
            Path to generated reward code
        """
        raise NotImplementedError("FM reward generator not implemented")
```

## Training Loop (Stub)

```python
def train_neural_rl_policy(
    curriculum_generator,
    initial_scenarios=["doorway_pinch", "dog_bed_soft"],
    max_iterations=1000,
    success_threshold=0.8
):
    """Train neural RL policy using FM-generated curriculum.
    
    Args:
        curriculum_generator: CurriculumGenerator instance
        initial_scenarios: Starting scenario set
        max_iterations: Max training iterations
        success_threshold: Success rate to advance curriculum
    
    Returns:
        Path to trained ONNX model
    """
    raise NotImplementedError("RL training loop not implemented")
```

## Sim-to-Real Transfer

**Reality gaps** (see `docs/sim-reality-gaps.md`):
- Sensor noise (depth, RGB, IMU)
- Dynamics (friction, wheel slip, soft contact)
- Lighting and appearance (texture, reflections)

**Mitigation**:
1. Domain randomization during training
2. Real-world fine-tuning via ASPIRE traces
3. Hybrid policy (neural + VFH fallback)

## Integration with ASPIRE

**Pre-training** (OMNI-EPIC):
1. FM generates diverse scenarios
2. Train neural policy in sim
3. Export ONNX → `models/policy.onnx`

**Deployment**:
4. Robot runs with neural_rl.py (+ VFH fallback)
5. trace_logger.py captures real-world execution

**Post-deployment refinement** (ASPIRE):
6. Analyze traces for failures
7. Coding agent updates skill library
8. Skills guide policy behavior or fallback strategies

## Future Work

**Not included in this scaffold**:
- FM curriculum generator implementation
- RL training pipeline (PPO/SAC)
- Host-sim integration for curriculum scenarios
- Domain randomization framework
- Sim-to-real transfer evaluation

**Priority**: Start with hand-coded scenarios in host-sim to validate neural_rl.py interface before adding FM curriculum generation.

## References

- **OMNI-EPIC**: arXiv:2405.15568 (Zhang, Faldor, Cully, Clune)
- **Host-sim**: `sim/` directory (Kevin 2D physics simulator)
- **Neural RL**: `src/neural_rl.py` (policy interface)
- **ASPIRE**: `SKILLS_ASPIRE.md` (skill refinement from traces)
