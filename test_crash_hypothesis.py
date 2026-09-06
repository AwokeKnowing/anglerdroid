"""Demonstrate the couch crash hypothesis.

Shows that forcing motion when fwd_scale=0 causes collisions.
"""

import sys
import numpy as np
from sim.world import create_scenario
from sim.robot import Robot
from sim.unsafe_policy import UnsafeCommitPolicy


def demonstrate_unsafe_commit():
    """Show that unsafe commit policy causes collision."""
    print("="*60)
    print("HYPOTHESIS TEST: Couch crash root cause")
    print("="*60)
    print()
    print("Testing: Does forcing motion when fwd_scale=0 cause collision?")
    print()
    
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    policy = UnsafeCommitPolicy()
    policy.reset()
    
    print("Starting position: robot facing couch")
    print()
    
    collision_detected = False
    collision_step = -1
    
    for step in range(200):
        robot.update_ego_maps(obs, height)
        
        safety_scales = {
            'fwd': robot.safety.fwd_scale,
            'bwd': robot.safety.bwd_scale,
            'ang': robot.safety.ang_scale,
        }
        
        pose = {'x': robot.x, 'y': robot.y, 'theta': robot.theta}
        
        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, safety_scales, pose)
        
        prev_x = robot.x
        robot.step(v_cmd, w_cmd, 0.033)
        
        actually_moved = abs(robot.x - prev_x) > 0.001
        
        if step % 20 == 0 or (step > 0 and step < 20):
            print(f"Step {step:3d}: "
                  f"fwd_scale={robot.safety.fwd_scale:.2f} "
                  f"v_cmd={v_cmd:.2f} "
                  f"v_actual={robot.v:.2f} "
                  f"moved={'YES' if actually_moved else 'NO '} "
                  f"mode={'COMMIT' if policy.commit_mode else 'cruise'}")
        
        if robot.check_collision() and not collision_detected:
            collision_detected = True
            collision_step = step
            print()
            print(f"⚠️  COLLISION DETECTED at step {step}")
            print(f"    Policy was in COMMIT mode: {policy.commit_mode}")
            print(f"    Safety fwd_scale: {robot.safety.fwd_scale:.2f}")
            print(f"    Robot actually moved: {actually_moved}")
            print()
            break
    
    print()
    print("="*60)
    print("RESULT:")
    print("="*60)
    
    if collision_detected:
        print()
        print("❌ HYPOTHESIS CONFIRMED")
        print()
        print("The unsafe commit policy caused a collision by forcing")
        print("forward motion (v=0.15 m/s) even when fwd_scale=0.")
        print()
        print("This matches the suspected behavior of the real couch crash:")
        print("  - Robot detected obstacle (fwd_scale → 0)")
        print("  - Recover/commit logic decided to 'commit' forward")
        print("  - Motion command overrode safety hard stop")
        print("  - Robot drove into couch ❌")
        print()
        print(f"Collision occurred at step {collision_step}")
        print()
        return 1
    else:
        print()
        print("⚠️  NO COLLISION - Policy did not override safety")
        print()
        return 0


if __name__ == '__main__':
    sys.exit(demonstrate_unsafe_commit())
