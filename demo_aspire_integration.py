#!/usr/bin/env python3
"""demo_aspire_integration.py — Quick demo of ASPIRE components working together.

Shows how trace logger, neural RL stub, and torque tools integrate
with existing Kevin architecture (local_executive, safety, odrivecan).

Run without hardware: python3 demo_aspire_integration.py
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import trace_logger
import neural_rl
import torque_tools


def demo_trace_logging():
    """Demo: trace logger captures execution events."""
    print("=== Demo 1: Trace Logger ===")
    
    # Initialize
    trace_logger.init("traces/demo_integration.jsonl", enable_rerun=False)
    
    # Simulate a navigation sequence
    print("Simulating navigation sequence...")
    
    # Start: goal set
    trace_logger.log_local_executive("xy", (2.5, 1.3), "vfh", True)
    
    # Normal motion
    trace_logger.log_twist_cmd(0.3, 0.1, safety_scaled=False)
    trace_logger.log_safety_event(None, 1.0, 1.0, 1.0, False)
    time.sleep(0.1)
    
    # Safety reflex triggered
    trace_logger.log_safety_event("topdown_near_field", 0.0, 0.8, 1.0, True)
    trace_logger.log_twist_cmd(0.0, 0.0, safety_scaled=True)
    time.sleep(0.1)
    
    # Skill invocation
    trace_logger.log_skill_invocation(
        "dog_bed_soft_approach",
        {"distance_cm": 65, "height_cm": 12},
        outcome="in_progress"
    )
    time.sleep(0.1)
    
    # Recovery
    trace_logger.log_safety_event(None, 1.0, 1.0, 1.0, False)
    trace_logger.log_twist_cmd(0.2, 0.0, safety_scaled=False)
    
    # Complete
    trace_logger.log_skill_invocation(
        "dog_bed_soft_approach",
        {"distance_cm": 65, "height_cm": 12},
        outcome="success"
    )
    
    trace_logger.shutdown()
    
    print("✅ Trace written to traces/demo_integration.jsonl")
    print()


def demo_neural_rl_policy():
    """Demo: neural RL policy with fallback."""
    print("=== Demo 2: Neural RL Policy ===")
    
    # Initialize policy (no model → fallback mode)
    policy = neural_rl.NeuralRLPolicy(
        model_path=None,
        inference_budget_ms=5.0,
        fallback_planner="vfh"
    )
    
    print(f"Policy initialized: {policy.get_debug_state()}")
    
    # Set goal
    policy.set_goal(2.5, 1.3)
    print(f"Goal set: {policy.get_debug_state()}")
    
    # Simulate tick
    obs_map = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    cmd = policy.tick(obs_map, pose, 0.033)
    
    if cmd is not None:
        print(f"Command: fwd={cmd['fwd_mps']:.2f} ang={cmd['ang_rads']:.2f} source={cmd['source']}")
    else:
        print("No command (fallback not implemented in stub)")
    
    print(f"Debug: {policy.get_debug_state()}")
    print("✅ Neural RL policy stub functional\n")


def demo_torque_tools():
    """Demo: torque tools with mock ODrive axes."""
    print("=== Demo 3: Torque Tools ===")
    
    # Create mock axes
    left = torque_tools.MockODriveAxis("Left")
    right = torque_tools.MockODriveAxis("Right")
    
    # Create controller
    controller = torque_tools.TorqueController(left, right)
    
    print("Controller initialized")
    
    # Micro-move straight
    print("Executing: micro_move_straight(5cm, 0.3Nm)")
    controller.micro_move_straight(0.05, force_nm=0.3, max_duration_s=0.2)
    print(f"  Left torque: {left.get_torque():.3f} Nm (should be 0 after stop)")
    
    # Micro-rotate
    print("Executing: micro_rotate(15°, 0.3Nm)")
    import math
    controller.micro_rotate(math.radians(15), torque_nm=0.3, max_duration_s=0.2)
    print(f"  Right torque: {right.get_torque():.3f} Nm (should be 0 after stop)")
    
    # Force-limited contact
    print("Executing: force_limited_contact(forward, 0.2Nm)")
    result = controller.force_limited_contact("forward", max_force_nm=0.2, duration_s=0.1)
    print(f"  Result: {result}")
    
    # Geometric conversion
    force_n = 10.0
    torque_nm = torque_tools.torque_from_force_linear(force_n)
    print(f"\nGeometric helper: {force_n}N → {torque_nm:.3f}Nm")
    
    print("✅ Torque tools functional\n")


def demo_complete_tick():
    """Demo: complete 30Hz tick with all components."""
    print("=== Demo 4: Complete Tick Integration ===")
    
    # Initialize components
    trace_logger.init("traces/demo_tick.jsonl", enable_rerun=False)
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    policy.set_goal(2.5, 1.3)
    
    print("Simulating 3 ticks of main loop...")
    
    for i in range(3):
        # Simulate obs map
        obs_map = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        pose = (0.0, 0.0, 0.0)
        
        # Neural RL tick
        cmd = policy.tick(obs_map, pose, 0.033)
        
        # Simulate safety state
        safety_state = {
            "reflex_type": None if i != 1 else "topdown_soft_low_obstacle",
            "fwd_scale": 1.0 if i != 1 else 0.3,
            "bwd_scale": 1.0,
            "ang_scale": 1.0,
            "throttled": i == 1
        }
        
        # Log complete tick
        trace_logger.log_tick_snapshot(
            twist_cmd=(0.3, 0.1) if cmd is None else (cmd['fwd_mps'], cmd['ang_rads']),
            safety_state=safety_state,
            executive_state=policy.get_debug_state(),
            obs_map_summary={
                "shape": obs_map.shape,
                "occupied_px": int((obs_map >= 100).sum())
            }
        )
        
        print(f"  Tick {i+1}: {'throttled' if safety_state['throttled'] else 'normal'}")
        time.sleep(0.033)
    
    trace_logger.shutdown()
    
    print("✅ Complete tick integration functional")
    print("   Trace: traces/demo_tick.jsonl\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ASPIRE Integration Demo")
    print("="*60 + "\n")
    
    demo_trace_logging()
    demo_neural_rl_policy()
    demo_torque_tools()
    demo_complete_tick()
    
    print("="*60)
    print("All demos completed successfully!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("  1. Review traces: cat traces/demo_*.jsonl | jq")
    print("  2. Integrate trace logging into main.py (optional)")
    print("  3. Deploy ONNX model for neural_rl (optional)")
    print("  4. Use torque_tools for calibration tasks")
    print("  5. Refine skills based on execution traces")
