"""
local_executive.py – Non-ROS mid-layer for autonomous navigation.

Accepts high-level goals (x,y) or "wander" mode and computes velocity commands
at ~30Hz using the current occupancy map and pose.

Phase 1: Minimal scaffold (API only, stub implementation)
Phase 2: DWA v0 (Dynamic Window Approach for local planning)
Phase 3: Wander mode (frontier-based exploration)
Phase 4: Neural upgrade (optional, learned local planner)

Units:
  - World frame: x=right, y=up, theta=0 facing +x (CCW), meters/radians
  - Ego frame: robot at (81, 119) facing RIGHT, 1px = 1cm
  - Commands: (fwd_mps, ang_rads) in m/s and rad/s
"""

import math
import time
import numpy as np
from robot_config import RCX, RCY, EGO_PX_SIZE, WHEELBASE_M


class LocalExecutive:
    """
    Mid-layer for autonomous navigation (non-ROS).
    
    Accepts goals in world frame, computes velocity commands from ego-space
    occupancy map. Designed to run at 30 Hz in the main loop.
    
    Current implementation: STUB (Phase 1)
    - API complete, minimal logic
    - Returns None (inactive) until DWA v0 is implemented
    
    TODO Phase 2: Implement DWA trajectory planning
    TODO Phase 3: Implement wander mode (frontier detection)
    TODO Phase 4: Neural planner upgrade (optional)
    """
    
    def __init__(self,
                 horizon_sec=1.0,
                 v_samples=7,
                 w_samples=9,
                 v_max=0.25,
                 w_max=0.8,
                 subgoal_dist=0.8,
                 min_clearance=0.20,
                 goal_tolerance=0.10):
        """
        Initialize LocalExecutive with DWA parameters.
        
        Args:
            horizon_sec: Trajectory lookahead time (seconds)
            v_samples: Number of linear velocity samples
            w_samples: Number of angular velocity samples
            v_max: Maximum linear velocity (m/s)
            w_max: Maximum angular velocity (rad/s)
            subgoal_dist: Rolling subgoal distance (meters)
            min_clearance: Minimum obstacle clearance (meters)
            goal_tolerance: Distance to goal for completion (meters)
        """
        # DWA parameters
        self.horizon_sec = horizon_sec
        self.v_samples = v_samples
        self.w_samples = w_samples
        self.v_max = v_max
        self.w_max = w_max
        self.subgoal_dist = subgoal_dist
        self.min_clearance = min_clearance
        self.goal_tolerance = goal_tolerance
        
        # State
        self._goal = None          # (x, y) in world frame (meters)
        self._wander_mode = False  # True if wandering (no explicit goal)
        self._active = False       # True if goal or wander mode set
        
        # Current velocity (for dynamic window)
        self._v_current = 0.0      # m/s
        self._w_current = 0.0      # rad/s
        
        # Debug state (for visualization)
        self._debug = {
            'goal_x': 0.0,
            'goal_y': 0.0,
            'subgoal_x': 0.0,
            'subgoal_y': 0.0,
            'cmd_fwd': 0.0,
            'cmd_ang': 0.0,
            'trajectories': [],  # List of (N, 3) arrays [x, y, theta]
            'best_traj_idx': -1,
        }
        
        print("LocalExecutive: initialized (STUB - Phase 1)")
        print("  DWA params: horizon=%.1fs v_samples=%d w_samples=%d" % (
            horizon_sec, v_samples, w_samples))
        print("  Limits: v_max=%.2f m/s w_max=%.2f rad/s" % (v_max, w_max))
        print("  NOTE: Currently returns None (inactive). Implement DWA for Phase 2.")
    
    def set_goal(self, x: float, y: float) -> None:
        """
        Set a world-space goal position (meters).
        The executive will drive toward rolling ~0.8-1.0m subgoals.
        
        Args:
            x: Goal x coordinate in world frame (meters)
            y: Goal y coordinate in world frame (meters)
        """
        self._goal = (float(x), float(y))
        self._wander_mode = False
        self._active = True
        print("LocalExecutive: goal set to (%.2f, %.2f) world frame" % (x, y))
    
    def set_wander_mode(self, enabled: bool) -> None:
        """
        Enable/disable wander mode (explore, avoid obstacles).
        Disables explicit goal if enabled.
        
        Args:
            enabled: True to enable wander mode, False to disable
        """
        self._wander_mode = enabled
        if enabled:
            self._goal = None
            self._active = True
            print("LocalExecutive: wander mode ENABLED")
        else:
            self._active = False
            print("LocalExecutive: wander mode DISABLED")
    
    def cancel(self) -> None:
        """
        Cancel current goal/wander mode.
        Next tick() will return None (zero velocity).
        """
        self._goal = None
        self._wander_mode = False
        self._active = False
        self._v_current = 0.0
        self._w_current = 0.0
        print("LocalExecutive: cancelled (inactive)")
    
    def is_active(self) -> bool:
        """Return True if goal or wander mode is set."""
        return self._active
    
    def tick(self, obs_map: np.ndarray, pose: tuple, dt: float) -> dict | None:
        """
        Compute velocity command for this frame.
        
        Args:
            obs_map: (H, W) uint8 ego-space obstacles (0=free, 1-100=height cm)
            pose: (x, y, theta) in world frame (meters, radians)
            dt: time since last tick (seconds)
        
        Returns:
            {'fwd_mps': float, 'ang_rads': float} or None if inactive.
            Caller applies safety scaling (already integrated in wheelbase).
        """
        if not self._active:
            return None
        
        x, y, theta = pose
        
        # STUB: Phase 1 - Just return None for now
        # TODO Phase 2: Implement DWA here
        #   1. Compute rolling subgoal (world or ego frame)
        #   2. Generate velocity samples (v, w) within dynamic window
        #   3. Simulate trajectories (forward integrate)
        #   4. Score trajectories (goal, obstacle, smoothness)
        #   5. Select best trajectory
        #   6. Return first (v, w) from best trajectory
        
        # Placeholder: check if goal reached
        if self._goal is not None:
            dx = self._goal[0] - x
            dy = self._goal[1] - y
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < self.goal_tolerance:
                print("LocalExecutive: goal reached (%.2f m), cancelling" % dist)
                self.cancel()
                return None
            
            # Update debug state
            self._debug['goal_x'] = self._goal[0]
            self._debug['goal_y'] = self._goal[1]
        
        # STUB: Return None (no command yet)
        # Once DWA is implemented, this will return {'fwd_mps': v, 'ang_rads': w}
        return None
    
    def get_debug_state(self) -> dict:
        """Return debug info for visualization (trajectories, subgoals, etc.)."""
        return self._debug.copy()
    
    # ── Helper methods for DWA (Phase 2) ──────────────────────────────────
    
    @staticmethod
    def _world_to_ego(dx_world, dy_world, theta):
        """
        Transform world-space vector to ego-space (robot-relative).
        
        Args:
            dx_world: x offset in world frame (meters)
            dy_world: y offset in world frame (meters)
            theta: robot heading in world frame (radians)
        
        Returns:
            (dx_ego, dy_ego) in ego frame (meters, robot facing +x)
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx_ego = dx_world * cos_t + dy_world * sin_t
        dy_ego = -dx_world * sin_t + dy_world * cos_t
        return dx_ego, dy_ego
    
    @staticmethod
    def _ego_to_world(dx_ego, dy_ego, theta):
        """
        Transform ego-space vector to world-space.
        
        Args:
            dx_ego: x offset in ego frame (meters, robot facing +x)
            dy_ego: y offset in ego frame (meters)
            theta: robot heading in world frame (radians)
        
        Returns:
            (dx_world, dy_world) in world frame (meters)
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx_world = dx_ego * cos_t - dy_ego * sin_t
        dy_world = dx_ego * sin_t + dy_ego * cos_t
        return dx_world, dy_world
    
    def _compute_subgoal(self, goal_world, pose):
        """
        Compute rolling subgoal (clamped to subgoal_dist ahead on path to goal).
        
        Args:
            goal_world: (x, y) in world frame (meters)
            pose: (x, y, theta) in world frame
        
        Returns:
            (subgoal_x, subgoal_y) in world frame (meters)
        """
        x, y, theta = pose
        dx = goal_world[0] - x
        dy = goal_world[1] - y
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < self.subgoal_dist:
            # Goal is close, use it directly
            return goal_world
        else:
            # Clamp to subgoal_dist ahead
            scale = self.subgoal_dist / dist
            return (x + dx * scale, y + dy * scale)
    
    def _generate_velocity_samples(self, dt):
        """
        Generate (v, w) samples within dynamic window.
        
        Args:
            dt: time step (seconds)
        
        Returns:
            List of (v, w) tuples (m/s, rad/s)
        
        TODO Phase 2: Implement dynamic window constraints
          - Reachable velocities given current (v, w) and acceleration limits
          - Admissible velocities (can stop before obstacle)
        """
        # STUB: Simple uniform grid (no dynamic constraints yet)
        v_samples = np.linspace(-self.v_max * 0.2, self.v_max, self.v_samples)
        w_samples = np.linspace(-self.w_max, self.w_max, self.w_samples)
        
        samples = []
        for v in v_samples:
            for w in w_samples:
                samples.append((v, w))
        return samples
    
    def _simulate_trajectory(self, v, w, pose, dt, steps):
        """
        Forward integrate trajectory from (v, w) for N steps.
        
        Args:
            v: linear velocity (m/s)
            w: angular velocity (rad/s)
            pose: (x, y, theta) starting pose
            dt: time step (seconds)
            steps: number of steps
        
        Returns:
            (N, 3) array [x, y, theta] in world frame
        """
        traj = np.zeros((steps, 3), dtype=np.float32)
        x, y, theta = pose
        
        for i in range(steps):
            # Differential drive kinematics
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += w * dt
            theta = math.atan2(math.sin(theta), math.cos(theta))  # Normalize
            traj[i] = [x, y, theta]
        
        return traj
    
    def _trajectory_collision_cost(self, traj, obs_map, pose):
        """
        Check trajectory for collisions in ego-space.
        
        Args:
            traj: (N, 3) array [x, y, theta] in world frame
            obs_map: (H, W) uint8 ego-space obstacles
            pose: (x, y, theta) current robot pose
        
        Returns:
            min_clearance_m: minimum clearance to obstacles (meters)
                            < min_clearance → high cost
        
        TODO Phase 2: Implement bounding circle collision check
        """
        # STUB: Return large clearance (no collision check yet)
        return 1.0  # meters
    
    def _score_trajectory(self, traj, goal_world, obs_map, pose):
        """
        Score trajectory for goal progress, obstacle clearance, smoothness.
        
        Args:
            traj: (N, 3) array [x, y, theta] in world frame
            goal_world: (x, y) goal in world frame (meters)
            obs_map: (H, W) uint8 ego-space obstacles
            pose: (x, y, theta) current robot pose
        
        Returns:
            score: float (higher is better)
        
        TODO Phase 2: Implement scoring function
          - goal_cost: distance to goal at end of trajectory
          - obstacle_cost: clearance penalty
          - smoothness_cost: angular/linear jerk
        """
        # STUB: Return zero score
        return 0.0


# ── Unit test / standalone demo ───────────────────────────────────────

if __name__ == "__main__":
    print("LocalExecutive unit test (Phase 1 stub)")
    
    # Create instance
    le = LocalExecutive()
    
    # Test API
    assert not le.is_active(), "Should start inactive"
    
    le.set_goal(2.0, 1.0)
    assert le.is_active(), "Should be active after set_goal"
    
    # Dummy obs_map and pose
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    # Tick (should return None in Phase 1)
    cmd = le.tick(obs_map, pose, 0.033)
    assert cmd is None, "Phase 1 stub should return None"
    
    le.cancel()
    assert not le.is_active(), "Should be inactive after cancel"
    
    le.set_wander_mode(True)
    assert le.is_active(), "Should be active in wander mode"
    
    cmd = le.tick(obs_map, pose, 0.033)
    assert cmd is None, "Wander mode stub should return None"
    
    le.cancel()
    
    # Test debug state
    dbg = le.get_debug_state()
    assert 'goal_x' in dbg, "Debug state should have goal_x"
    
    print("✓ All unit tests passed (Phase 1 stub)")
