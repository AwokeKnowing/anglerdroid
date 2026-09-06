"""CLI interface for running simulator scenarios.

Usage:
    python -m sim.run --steps N --scenario couch_pinch --save out.gif
"""

import argparse
import math
import sys
import numpy as np

from sim.world import create_scenario, FRAME_W, FRAME_H, RCX, RCY, EGO_PX_SIZE
from sim.robot import Robot
from sim.policy import create_policy
from sim.metrics import run_episode as metrics_run_episode


def run_simulation(scenario_name, policy_name, steps, dt, save_path=None, render=False):
    """Run a simulation episode.
    
    Returns:
        metrics dict with collisions, time_to_clear, recover_counts, etc.
    """
    world_obs, world_height = create_scenario(scenario_name)
    
    start_x = RCX * EGO_PX_SIZE
    start_y = RCY * EGO_PX_SIZE
    start_theta = 0.0
    
    robot = Robot(start_x, start_y, start_theta)
    policy = create_policy(policy_name)
    policy.reset()
    
    frames = []
    collision_detected = False
    collision_step = -1
    
    for step in range(steps):
        robot.update_ego_maps(world_obs, world_height)
        
        if robot.check_collision():
            if not collision_detected:
                collision_detected = True
                collision_step = step
                robot.collision_count += 1
        
        safety_scales = {
            'fwd': robot.safety.fwd_scale,
            'bwd': robot.safety.bwd_scale,
            'ang': robot.safety.ang_scale,
        }
        
        pose = {
            'x': robot.x,
            'y': robot.y,
            'theta': robot.theta,
        }
        
        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, safety_scales, pose)
        
        robot.step(v_cmd, w_cmd, dt)
        
        if render or save_path:
            frame = render_frame(robot.ego_obs, robot, step, safety_scales)
            frames.append(frame)
        
        if step % 50 == 0:
            print(f"Step {step}/{steps}: pos=({robot.x:.2f}, {robot.y:.2f}) "
                  f"theta={math.degrees(robot.theta):.1f}° "
                  f"safety=(f:{robot.safety.fwd_scale:.2f} b:{robot.safety.bwd_scale:.2f} a:{robot.safety.ang_scale:.2f}) "
                  f"collisions={robot.collision_count}")
    
    if save_path and frames:
        save_frames(frames, save_path)
    
    metrics = {
        'collisions': robot.collision_count,
        'collision_step': collision_step,
        'final_x': robot.x,
        'final_y': robot.y,
        'final_theta': robot.theta,
    }
    
    return metrics


def render_frame(ego_obs, robot, step, safety_scales):
    """Render a single frame for visualization."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    
    frame[:, :, 0] = ego_obs
    frame[:, :, 1] = ego_obs
    frame[:, :, 2] = ego_obs
    
    fwd_color = int(255 * safety_scales['fwd'])
    bwd_color = int(255 * safety_scales['bwd'])
    ang_color = int(255 * safety_scales['ang'])
    
    frame[0:10, 0:50] = [fwd_color, 0, 0]
    frame[0:10, 50:100] = [0, bwd_color, 0]
    frame[0:10, 100:150] = [0, 0, ang_color]
    
    cv2_available = False
    try:
        import cv2
        cv2_available = True
    except ImportError:
        pass
    
    if cv2_available:
        import cv2
        cv2.circle(frame, (RCX, RCY), 3, (0, 255, 0), -1)
        
        fwd_x = RCX + 15
        fwd_y = RCY
        cv2.arrowedLine(frame, (RCX, RCY), (fwd_x, fwd_y), (255, 255, 0), 2)
    
    return frame


def save_frames(frames, save_path):
    """Save frames as GIF or PNG strip."""
    if save_path.endswith('.gif'):
        try:
            import imageio
            imageio.mimsave(save_path, frames, fps=30, loop=0)
            print(f"Saved GIF: {save_path}")
        except ImportError:
            print("Warning: imageio not available, saving first frame as PNG")
            save_png_strip(frames, save_path.replace('.gif', '.png'))
    else:
        save_png_strip(frames, save_path)


def save_png_strip(frames, save_path):
    """Save frames as horizontal PNG strip."""
    try:
        import imageio
        if len(frames) > 10:
            step = len(frames) // 10
            frames = frames[::step]
        
        strip = np.hstack(frames)
        imageio.imwrite(save_path, strip)
        print(f"Saved PNG strip: {save_path}")
    except ImportError:
        print("Warning: imageio not available, cannot save PNG")


def main():
    parser = argparse.ArgumentParser(description='Run lightweight 2D simulator')
    parser.add_argument('--steps', type=int, default=200, help='Number of simulation steps')
    parser.add_argument('--scenario', type=str, default='empty', 
                        choices=['empty', 'couch_pinch', 'house', 'hallway', 'doorway', 'l_corner'],
                        help='Scenario to run')
    parser.add_argument('--policy', type=str, default='housebot',
                        choices=['random', 'housebot', 'stop', 'unsafe'],
                        help='Policy to use')
    parser.add_argument('--hz', type=float, default=30.0, help='Simulation frequency (Hz)')
    parser.add_argument('--save', type=str, default=None, help='Save path for GIF or PNG')
    parser.add_argument('--render', action='store_true', help='Render frames (requires save or display)')
    
    args = parser.parse_args()
    
    dt = 1.0 / args.hz
    
    print(f"Running simulation: scenario={args.scenario} policy={args.policy} steps={args.steps} hz={args.hz}")
    print(f"Timestep dt={dt:.4f}s")
    
    if args.save or args.render:
        metrics = run_simulation(
            args.scenario,
            args.policy,
            args.steps,
            dt,
            save_path=args.save,
            render=args.render
        )
    else:
        metrics = metrics_run_episode(
            args.scenario, args.policy, steps=args.steps, dt=dt
        )
    
    print("\n=== SIMULATION COMPLETE ===")
    print(f"Collisions: {metrics['collisions']}")
    if metrics['collision_step'] >= 0:
        print(f"First collision at step: {metrics['collision_step']}")
    print(f"Final pose: x={metrics['final_x']:.3f}m y={metrics['final_y']:.3f}m theta={math.degrees(metrics['final_theta']):.1f}°")
    if args.scenario == "doorway" and "doorway_crossed" in metrics:
        print(
            f"Doorway crossed: {metrics['doorway_crossed']} "
            f"(step={metrics.get('doorway_crossed_step', -1)}) "
            f"path_m={metrics.get('path_len_m', 0):.2f}"
        )
    
    if metrics['collisions'] > 0:
        print("\n❌ FAILED: Collision detected")
        sys.exit(1)
    else:
        print("\n✅ PASSED: No collisions")
        sys.exit(0)


if __name__ == '__main__':
    main()
