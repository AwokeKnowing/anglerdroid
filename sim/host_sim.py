"""High-fidelity host simulator entrypoint (i777).

Drop-in loop running SafetyGuard + MPPI + house_bot stubs at 30Hz.
Designed for transfer learning / offline iteration on RTX 3090.

Usage:
    python -m sim.host_sim --scenario table_overhang --duration 30 --fidelity high --save demo.gif
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for live stack imports
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Simulator components
from sim.sensors import SensorSuite, SimClock
from sim.world_enhanced import (
    create_enhanced_scenario,
    scenario_start_enhanced,
    get_soft_cost_scale,
)
from sim.robot import Robot, SafetyGuard
from sim.dynamics import DiffDriveDynamics
from sim.mppi_policy import MppiSimPolicy
from sim.viz import render_world_frame

# Live stack imports (drop-in compatibility)
try:
    from robot_config import (
        RCX, RCY, EGO_PX_SIZE, FRAME_W, FRAME_H,
        FOOT_X0, FOOT_Y0, FOOT_X1, FOOT_Y1,
    )
except ImportError:
    # Fallback if robot_config not found
    FRAME_W, FRAME_H = 320, 240
    RCX, RCY = 81, 119
    EGO_PX_SIZE = 0.01
    FOOT_X0, FOOT_Y0, FOOT_X1, FOOT_Y1 = 56, 97, 101, 142


class HostSimulator:
    """High-fidelity host simulator for Kevin's autonomy stack."""
    
    def __init__(
        self,
        scenario='table_overhang',
        fidelity='medium',
        hz=30.0,
        policy_name='mppi',
        use_soft_cost=True,
        seed=None,
        rerun_enabled=False,
    ):
        """Initialize simulator.
        
        Args:
            scenario: World scenario name
            fidelity: 'none', 'low', 'medium', 'high'
            hz: Simulation frequency (Hz)
            policy_name: 'mppi', 'housebot', etc.
            use_soft_cost: Enable MPPI soft prefer costs
            seed: Random seed for reproducibility
            rerun_enabled: Enable Rerun logging
        """
        self.scenario = scenario
        self.fidelity = fidelity
        self.hz = hz
        self.dt = 1.0 / hz
        self.seed = seed
        
        # World + metadata
        self.world_obs, self.world_height, self.metadata = create_enhanced_scenario(scenario)
        
        # Start pose
        start_x, start_y, start_theta = scenario_start_enhanced(scenario)
        
        # Robot with fidelity dynamics
        fidelity_dyn = (fidelity != 'none')
        self.robot = Robot(start_x, start_y, start_theta, fidelity=fidelity_dyn)
        
        # Sensors
        self.sensors = SensorSuite(fidelity=fidelity, seed=seed)
        
        # Safety guard (matches src/safety.py)
        self.safety = SafetyGuard()
        
        # Policy
        if policy_name == 'mppi':
            self.policy = MppiSimPolicy(wander=True, use_soft_cost=use_soft_cost)
        else:
            # Fall back to simple policy
            from sim.policy import create_policy
            self.policy = create_policy(policy_name, use_soft_cost=use_soft_cost)
        self.policy.reset()
        
        # Clock
        self.clock = SimClock(hz=hz, real_time_factor=1.0)
        
        # Rerun logging
        self.rerun_enabled = rerun_enabled
        if rerun_enabled:
            try:
                from sim.rerun_logger import SimRerunLogger
                self.rerun_logger = SimRerunLogger(f"sim_{scenario}")
            except ImportError:
                print("Warning: Rerun not available, logging disabled")
                self.rerun_enabled = False
        
        # Metrics
        self.step = 0
        self.collision_count = 0
        self.collision_step = -1
        self.trail = []
        self.frames = []
        
    def run(self, duration_sec, save_path=None):
        """Run simulation for specified duration.
        
        Args:
            duration_sec: Simulation time (seconds)
            save_path: Optional path to save visualization GIF
            
        Returns:
            metrics: dict with results
        """
        max_steps = int(duration_sec * self.hz)
        
        print(f"Running {self.scenario} for {duration_sec}s ({max_steps} steps @ {self.hz}Hz)")
        print(f"Fidelity: {self.fidelity}, Policy: {self.policy.__class__.__name__}")
        
        wall_start = time.monotonic()
        
        for self.step in range(max_steps):
            # 1. Update ego maps from sensors
            self.robot.update_ego_maps(self.world_obs, self.world_height)
            
            # 2. Sensor simulation (add noise to ego maps if fidelity enabled)
            if self.fidelity != 'none':
                obs_ego, height_ego = self.sensors.capture_depth(
                    self._get_pose_dict(),
                    self.world_obs,
                    self.world_height,
                )
                # Override robot's ego maps with noisy sensor readings
                self.robot.ego_obs = obs_ego
                self.robot.ego_height = height_ego
                # Re-run safety on noisy obs
                self.robot.safety.update(obs_ego)
            
            # 3. Check collision
            if self.robot.check_collision():
                if self.collision_step < 0:
                    self.collision_step = self.step
                    self.collision_count += 1
            
            # 4. Get safety scales
            safety_scales = self._get_safety_scales()
            
            # 5. Policy decision
            pose = self._get_pose_dict()
            v_cmd, w_cmd = self.policy.act(
                self.robot.ego_obs,
                self.robot.ego_height,
                safety_scales,
                pose,
            )
            
            # 6. Step robot dynamics
            self.robot.step(v_cmd, w_cmd, self.dt, apply_safety=True)
            
            # 7. Record trail
            self.trail.append((self.robot.x, self.robot.y))
            
            # 8. Render frame (optional)
            if save_path and self.step % 3 == 0:
                frame = render_world_frame(
                    self.world_obs,
                    self.robot,
                    self.step,
                    safety_scales,
                    self.trail,
                )
                self.frames.append(frame)
            
            # 9. Rerun logging (optional)
            if self.rerun_enabled and self.step % 6 == 0:
                self._log_rerun_frame(obs_ego if self.fidelity != 'none' else self.robot.ego_obs)
            
            # 10. Progress print
            if self.step % 30 == 0 and self.step > 0:
                elapsed = time.monotonic() - wall_start
                sim_time = self.step * self.dt
                rt_factor = sim_time / elapsed if elapsed > 0 else 0
                print(
                    f"  Step {self.step}/{max_steps}: "
                    f"pos=({self.robot.x:.2f}, {self.robot.y:.2f}) "
                    f"θ={math.degrees(self.robot.theta):.1f}° "
                    f"v={self.robot.v:.2f} w={self.robot.w:.2f} "
                    f"safety=(f:{safety_scales['fwd']:.2f} b:{safety_scales['bwd']:.2f} a:{safety_scales['ang']:.2f}) "
                    f"RT×{rt_factor:.1f}"
                )
            
            # 11. Clock tick
            self.clock.tick()
        
        # Save visualization
        if save_path and self.frames:
            self._save_gif(save_path)
        
        # Compute metrics
        wall_elapsed = time.monotonic() - wall_start
        metrics = self._compute_metrics(wall_elapsed)
        
        return metrics
    
    def _get_pose_dict(self):
        """Get robot pose as dict."""
        return {
            'x': self.robot.x,
            'y': self.robot.y,
            'theta': self.robot.theta,
        }
    
    def _get_safety_scales(self):
        """Get safety scales dict (matches live format)."""
        return {
            'fwd': self.robot.safety.fwd_scale,
            'bwd': self.robot.safety.bwd_scale,
            'ang': self.robot.safety.ang_scale,
            'fwd_m': float(self.robot.safety.fwd_clear) * EGO_PX_SIZE,
            'bwd_m': float(self.robot.safety.bwd_clear) * EGO_PX_SIZE,
            'lat_m': float(self.robot.safety.lat_clear) * EGO_PX_SIZE,
        }
    
    def _log_rerun_frame(self, obs_ego):
        """Log frame to Rerun (stub for now)."""
        pass  # TODO: Implement Rerun logging
    
    def _save_gif(self, path):
        """Save frames as GIF."""
        try:
            import imageio
            fps = 10 if len(self.frames) > 100 else 20
            imageio.mimsave(path, self.frames, fps=fps, loop=0)
            print(f"Saved GIF: {path} ({len(self.frames)} frames)")
        except ImportError:
            print("Warning: imageio not available, cannot save GIF")
    
    def _compute_metrics(self, wall_elapsed):
        """Compute final metrics."""
        sim_time = self.step * self.dt
        rt_factor = sim_time / wall_elapsed if wall_elapsed > 0 else 0
        
        metrics = {
            'scenario': self.scenario,
            'fidelity': self.fidelity,
            'steps': self.step,
            'sim_time_sec': sim_time,
            'wall_time_sec': wall_elapsed,
            'real_time_factor': rt_factor,
            'collisions': self.collision_count,
            'collision_step': self.collision_step,
            'final_x': self.robot.x,
            'final_y': self.robot.y,
            'final_theta': self.robot.theta,
            'seed': self.seed,
        }
        
        return metrics


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='High-fidelity host simulator for Kevin (i777)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Table overhang stress test (30 seconds)
  python -m sim.host_sim --scenario table_overhang --duration 30 --fidelity high --save table.gif
  
  # Soft bed scoring
  python -m sim.host_sim --scenario soft_bed --duration 60 --fidelity medium
  
  # Full house wander (5 minutes)
  python -m sim.host_sim --scenario combined_house --duration 300 --fidelity low --seed 42
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='table_overhang',
        choices=[
            'table_overhang',
            'table_overhang_tight',
            'soft_bed',
            'keepout_mat',
            'tight_doorway',
            'hallway_pinch',
            'combined_house',
            # Base scenarios from sim/world.py
            'empty',
            'couch_pinch',
            'house',
            'hallway',
            'doorway',
            'cul_de_sac',
        ],
        help='Scenario to run',
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Simulation duration (seconds)',
    )
    
    parser.add_argument(
        '--fidelity',
        type=str,
        default='medium',
        choices=['none', 'low', 'medium', 'high'],
        help='Fidelity mode (sensor noise, latency, drift)',
    )
    
    parser.add_argument(
        '--hz',
        type=float,
        default=30.0,
        help='Simulation frequency (Hz)',
    )
    
    parser.add_argument(
        '--policy',
        type=str,
        default='mppi',
        choices=['mppi', 'housebot', 'goalseek', 'random', 'stop'],
        help='Policy to use',
    )
    
    parser.add_argument(
        '--soft-cost',
        dest='soft_cost',
        action='store_true',
        default=True,
        help='Enable MPPI soft prefer costs (default ON)',
    )
    
    parser.add_argument(
        '--no-soft-cost',
        dest='soft_cost',
        action='store_false',
        help='Disable MPPI soft prefer costs',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility',
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save visualization GIF to path',
    )
    
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='Enable Rerun logging',
    )
    
    args = parser.parse_args()
    
    # Auto-save to artifacts if --save omitted
    if args.save is None and args.duration <= 60:
        artifacts_dir = Path(__file__).parents[1] / 'artifacts' / 'sim'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        args.save = str(artifacts_dir / f'{args.scenario}_{args.fidelity}.gif')
        print(f"Auto-save enabled: {args.save}")
    
    # Create simulator
    sim = HostSimulator(
        scenario=args.scenario,
        fidelity=args.fidelity,
        hz=args.hz,
        policy_name=args.policy,
        use_soft_cost=args.soft_cost,
        seed=args.seed,
        rerun_enabled=args.rerun,
    )
    
    # Run
    metrics = sim.run(duration_sec=args.duration, save_path=args.save)
    
    # Print results
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Scenario: {metrics['scenario']}")
    print(f"Fidelity: {metrics['fidelity']}")
    print(f"Steps: {metrics['steps']} ({metrics['sim_time_sec']:.1f}s sim time)")
    print(f"Wall time: {metrics['wall_time_sec']:.1f}s (×{metrics['real_time_factor']:.1f} real-time)")
    print(f"Final pose: ({metrics['final_x']:.2f}, {metrics['final_y']:.2f}) θ={math.degrees(metrics['final_theta']):.1f}°")
    print(f"Collisions: {metrics['collisions']}")
    
    if metrics['collision_step'] >= 0:
        print(f"  First collision at step {metrics['collision_step']}")
    
    # Exit code
    if metrics['collisions'] > 0:
        print("\n❌ FAILED: Collision detected")
        sys.exit(1)
    else:
        print("\n✅ PASSED: No collisions")
        sys.exit(0)


if __name__ == '__main__':
    main()
