"""mppi_costmap.py – Phase-1 MPPI local planner stubs (mppi-costmap-v0).

Brought over from cloud-agent branch (PR #3) without waiting on merge.
Live robot currently drives via local_executive.py → VFH; this module is the
CUDA/Torch upgrade path behind the same mid-layer mailbox.

Run unit tests: python3 -m mppi_costmap
"""
import math
import time
import numpy as np
from robot_config import RCX, RCY, EGO_PX_SIZE, WHEELBASE_M


class MppiCostmapPlanner:
    """
    MPPI-based local planner for autonomous navigation (non-ROS).
    
    Accepts async map-frame (x,y) goals, computes velocity commands from 
    ego-space 2D costmap. Non-blocking mailbox design.
    
    Current implementation: STUB (Phase 1 scaffold)
    - API complete, mailbox pattern ready
    - Simple geometric placeholder (pure pursuit or similar)
    - MPPI hooks clearly marked for CUDA/Torch implementation
    
    MPPI Integration Points (see _mppi_sample_trajectories, _mppi_rollout_cost):
      - CUDA MPPI-Generic (cuMPPI on GitHub)
      - PyTorch MPPI (torch-mppi or custom batched rollout)
      - Cost function: obstacle + goal + smoothness on costmap window
    
    Future Phases:
      Phase 2: iPlanner-IL (imitation learning over MPPI) or SAC-polar
      Phase 3: ViNT/NoMaD (visual policies) + Doctor (safety filter over MPPI)
    """
    
    def __init__(self,
                 horizon_sec=1.0,
                 n_samples=512,
                 v_max=0.25,
                 w_max=0.8,
                 subgoal_dist=0.8,
                 min_clearance=0.20,
                 goal_tolerance=0.10,
                 mppi_temperature=1.0,
                 mppi_noise_sigma_v=0.1,
                 mppi_noise_sigma_w=0.3):
        """
        Initialize MppiCostmapPlanner with MPPI parameters.
        
        Args:
            horizon_sec: Trajectory lookahead time (seconds)
            n_samples: Number of MPPI trajectory samples (512-2048 typical)
            v_max: Maximum linear velocity (m/s)
            w_max: Maximum angular velocity (rad/s)
            subgoal_dist: Rolling subgoal distance (meters)
            min_clearance: Minimum obstacle clearance (meters)
            goal_tolerance: Distance to goal for completion (meters)
            mppi_temperature: Temperature parameter λ (lower = more aggressive)
            mppi_noise_sigma_v: Process noise std for linear velocity
            mppi_noise_sigma_w: Process noise std for angular velocity
        """
        # MPPI parameters
        self.horizon_sec = horizon_sec
        self.n_samples = n_samples
        self.dt = 0.033  # 30 Hz main loop
        self.n_steps = int(horizon_sec / self.dt)  # Typically 30 steps for 1s
        
        self.v_max = v_max
        self.w_max = w_max
        self.subgoal_dist = subgoal_dist
        self.min_clearance = min_clearance
        self.goal_tolerance = goal_tolerance
        
        # MPPI specific
        self.temperature = mppi_temperature
        self.noise_sigma_v = mppi_noise_sigma_v
        self.noise_sigma_w = mppi_noise_sigma_w
        
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
        
        print("MppiCostmapPlanner: initialized (mppi-costmap-v0, Phase 1 STUB)")
        print("  MPPI params: horizon=%.1fs n_samples=%d n_steps=%d" % (
            horizon_sec, n_samples, self.n_steps))
        print("  Limits: v_max=%.2f m/s w_max=%.2f rad/s" % (v_max, w_max))
        print("  Temperature λ=%.2f, noise σ_v=%.2f σ_ω=%.2f" % (
            mppi_temperature, mppi_noise_sigma_v, mppi_noise_sigma_w))
        print("  NOTE: Stub returns None. Implement MPPI sampling for full control.")
        print("  MPPI integration: see _mppi_sample_trajectories() for CUDA/Torch hook")
    
    def set_goal(self, x: float, y: float) -> None:
        """
        Set a map-frame goal position (meters). Non-blocking mailbox write.
        The executive will drive toward rolling ~0.8-1.0m subgoals.
        
        Thread-safe: Called from main loop or UI thread. Vision/capture never waits.
        
        Args:
            x: Goal x coordinate in map frame (meters)
            y: Goal y coordinate in map frame (meters)
        """
        self._goal = (float(x), float(y))
        self._wander_mode = False
        self._active = True
        print("MppiCostmapPlanner: goal set to (%.2f, %.2f) map frame" % (x, y))
    
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
            print("MppiCostmapPlanner: wander mode ENABLED")
        else:
            self._active = False
            print("MppiCostmapPlanner: wander mode DISABLED")
    
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
        print("MppiCostmapPlanner: cancelled (inactive)")
    
    def is_active(self) -> bool:
        """Return True if goal or wander mode is set."""
        return self._active
    
    def tick(self, obs_map: np.ndarray, pose: tuple, dt: float) -> dict | None:
        """
        Compute velocity command for this frame (non-blocking).
        
        Main loop calls this at 30Hz. Never blocks capture/vision thread.
        Reads latest goal from mailbox, runs MPPI on costmap window, returns cmd.
        
        Args:
            obs_map: (H, W) uint8 ego-space obstacles (0=free, 1-100=height cm)
                     Includes projected global map + fresh sensor data
            pose: (x, y, theta) in map frame (meters, radians)
            dt: time since last tick (seconds, typically 0.033)
        
        Returns:
            {'fwd_mps': float, 'ang_rads': float} or None if inactive.
            Caller applies safety scaling (already integrated in wheelbase).
        """
        if not self._active:
            return None
        
        x, y, theta = pose
        
        # STUB Phase 1: Geometric placeholder (e.g., pure pursuit)
        # TODO: Replace with MPPI sampling when ready
        #
        # MPPI Implementation Outline:
        #   1. Compute rolling subgoal (0.8-1.0m ahead on path to goal)
        #   2. Sample control sequences: U[k] ~ N(u_prev, Σ) for k=0..n_steps
        #      - CUDA hook: cuMPPI.sample_controls(n_samples, n_steps, noise_sigma)
        #      - Torch hook: torch.randn(...) * noise_sigma + u_prev
        #   3. Rollout trajectories: diff-drive kinematics on costmap
        #      - CUDA hook: cuMPPI.rollout_batch(U, x0, costmap_tex)
        #      - Torch hook: batched forward integration + costmap lookup
        #   4. Compute costs: S[i] = ∑ running_cost(x[k], u[k]) + terminal_cost(x[T])
        #      - Obstacle: collision penalty from costmap (lookup at trajectory points)
        #      - Goal: distance to subgoal at trajectory end
        #      - Smoothness: angular/linear acceleration magnitude
        #   5. Weight samples: w[i] = exp(-S[i] / λ), normalize
        #   6. Return weighted mean control: u* = ∑ w[i] * U[i,0]
        #
        # See _mppi_sample_trajectories(), _mppi_rollout_cost() below for stubs.
        
        # Placeholder: check if goal reached
        if self._goal is not None:
            dx = self._goal[0] - x
            dy = self._goal[1] - y
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < self.goal_tolerance:
                print("MppiCostmapPlanner: goal reached (%.2f m), cancelling" % dist)
                self.cancel()
                return None
            
            # Update debug state
            self._debug['goal_x'] = self._goal[0]
            self._debug['goal_y'] = self._goal[1]
            
            # STUB: Simple geometric controller (pure pursuit style)
            # Replace with MPPI when ready
            cmd = self._geometric_controller_stub(obs_map, pose)
            if cmd is not None:
                self._v_current = cmd['fwd_mps']
                self._w_current = cmd['ang_rads']
                self._debug['cmd_fwd'] = cmd['fwd_mps']
                self._debug['cmd_ang'] = cmd['ang_rads']
                return cmd
        
        # Wander mode stub (frontier-based, Phase 2)
        if self._wander_mode:
            # TODO: Detect frontiers, set virtual goal, call MPPI
            pass
        
        return None
    
    def get_debug_state(self) -> dict:
        """Return debug info for visualization (trajectories, subgoals, etc.)."""
        return self._debug.copy()
    
    # ── STUB: Geometric controller (replace with MPPI) ────────────────────
    
    def _geometric_controller_stub(self, obs_map, pose):
        """
        Simple pure-pursuit style controller for testing API.
        Replace with MPPI when ready.
        
        Returns {'fwd_mps': float, 'ang_rads': float} or None if blocked.
        """
        if self._goal is None:
            return None
        
        x, y, theta = pose
        
        # Compute ego-frame goal direction
        dx_world = self._goal[0] - x
        dy_world = self._goal[1] - y
        dx_ego, dy_ego = self._world_to_ego(dx_world, dy_world, theta)
        
        # Simple proportional control
        # Heading error in ego frame (atan2 of ego goal)
        heading_error = math.atan2(dy_ego, dx_ego)
        
        # Distance to goal
        dist = math.sqrt(dx_ego ** 2 + dy_ego ** 2)
        
        # Proportional gains (tunable)
        k_v = 0.3  # linear velocity gain
        k_w = 1.2  # angular velocity gain
        
        # Compute commands
        fwd = min(self.v_max, k_v * dist)
        ang = np.clip(k_w * heading_error, -self.w_max, self.w_max)
        
        # Slow down for large heading errors (like VFH)
        if abs(heading_error) > math.radians(30):
            fwd *= 0.5
        if abs(heading_error) > math.radians(60):
            fwd = 0.0  # Pure rotation
        
        # Simple obstacle check (stop if obstacle directly ahead)
        # TODO: Replace with MPPI cost function
        h, w = obs_map.shape
        rcx, rcy = RCX, RCY
        forward_strip = obs_map[rcy-10:rcy+10, rcx+10:min(w, rcx+40)]
        if forward_strip.max() > 100:  # Obstacle ahead
            fwd = 0.0
        
        return {'fwd_mps': fwd, 'ang_rads': ang}
    
    # ── MPPI Implementation Hooks (Phase 1 - to be filled in) ─────────────
    
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
    
    def _mppi_sample_trajectories(self, u_prev, n_samples, n_steps):
        """
        MPPI Step 1: Sample control sequences with noise.
        
        U[i,k] = u_prev[k] + ε[i,k], where ε ~ N(0, Σ)
        
        Args:
            u_prev: (n_steps, 2) array of previous control sequence [(v, w), ...]
            n_samples: Number of MPPI samples (512-2048)
            n_steps: Horizon length in steps
        
        Returns:
            U: (n_samples, n_steps, 2) array of control sequences
        
        CUDA Integration:
            Replace with cuMPPI.sample_controls(n_samples, n_steps, noise_sigma)
            or PyTorch: torch.randn(n_samples, n_steps, 2) * noise_sigma + u_prev
        """
        # STUB: Numpy placeholder
        U = np.zeros((n_samples, n_steps, 2), dtype=np.float32)
        for i in range(n_samples):
            noise_v = np.random.normal(0, self.noise_sigma_v, n_steps)
            noise_w = np.random.normal(0, self.noise_sigma_w, n_steps)
            U[i, :, 0] = u_prev[:, 0] + noise_v
            U[i, :, 1] = u_prev[:, 1] + noise_w
            # Clip to limits
            U[i, :, 0] = np.clip(U[i, :, 0], -self.v_max * 0.2, self.v_max)
            U[i, :, 1] = np.clip(U[i, :, 1], -self.w_max, self.w_max)
        return U
    
    def _mppi_rollout_batch(self, U, x0, obs_map):
        """
        MPPI Step 2 & 3: Rollout trajectories and compute costs.
        
        For each control sequence U[i], integrate diff-drive kinematics
        and accumulate cost from costmap + goal distance + smoothness.
        
        Args:
            U: (n_samples, n_steps, 2) control sequences [(v, w), ...]
            x0: (x, y, theta) starting pose in map frame
            obs_map: (H, W) uint8 ego-space costmap
        
        Returns:
            S: (n_samples,) array of trajectory costs (lower is better)
            X: (n_samples, n_steps, 3) array of trajectories [(x,y,θ), ...]
        
        CUDA Integration:
            Replace with cuMPPI.rollout_batch(U, x0, costmap_texture)
            or PyTorch: batched integration + torch costmap interpolation
        """
        n_samples, n_steps, _ = U.shape
        X = np.zeros((n_samples, n_steps, 3), dtype=np.float32)
        S = np.zeros(n_samples, dtype=np.float32)
        
        # STUB: Serial rollout (replace with parallel CUDA/Torch)
        for i in range(n_samples):
            x, y, theta = x0
            cost = 0.0
            
            for k in range(n_steps):
                v, w = U[i, k]
                
                # Integrate (differential drive)
                x += v * math.cos(theta) * self.dt
                y += v * math.sin(theta) * self.dt
                theta += w * self.dt
                theta = math.atan2(math.sin(theta), math.cos(theta))
                
                X[i, k] = [x, y, theta]
                
                # Running cost (stub - replace with proper cost function)
                cost += self._mppi_running_cost((x, y, theta), (v, w), obs_map, x0)
            
            # Terminal cost
            cost += self._mppi_terminal_cost(X[i, -1], x0)
            S[i] = cost
        
        return S, X
    
    def _mppi_running_cost(self, state, control, obs_map, x0):
        """
        MPPI running cost at step k.
        
        Cost = w_obs × obstacle_penalty + w_smooth × control_effort
        
        Obstacle penalty: Sample costmap at (x, y) in ego frame.
        Control effort: Penalize large |v|, |w|, |dv/dt|, |dw/dt|.
        """
        x, y, theta = state
        v, w = control
        
        # Transform map position to ego-space pixel
        dx_world = x - x0[0]
        dy_world = y - x0[1]
        dx_ego, dy_ego = self._world_to_ego(dx_world, dy_world, x0[2])
        
        # Ego pixel coordinates
        col = int(RCX + dx_ego / EGO_PX_SIZE)
        row = int(RCY - dy_ego / EGO_PX_SIZE)
        
        # Obstacle cost (sample costmap)
        obs_cost = 0.0
        h, w_map = obs_map.shape
        if 0 <= row < h and 0 <= col < w_map:
            obs_val = obs_map[row, col]
            if obs_val > 100:  # Obstacle
                obs_cost = 100.0  # High penalty
            elif obs_val > 50:
                obs_cost = obs_val / 100.0
        else:
            obs_cost = 10.0  # Out of costmap bounds
        
        # Smoothness cost (penalize large controls)
        smooth_cost = 0.01 * (v ** 2 + w ** 2)
        
        return obs_cost + smooth_cost
    
    def _mppi_terminal_cost(self, state, x0):
        """
        MPPI terminal cost at step T (end of horizon).
        
        Cost = w_goal × distance_to_subgoal
        """
        if self._goal is None:
            return 0.0
        
        x, y, theta = state
        
        # Compute subgoal (rolling 0.8-1.0m ahead)
        subgoal = self._compute_subgoal(self._goal, x0)
        
        # Distance to subgoal
        dx = subgoal[0] - x
        dy = subgoal[1] - y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        
        return 10.0 * dist  # Weight: prefer ending closer to subgoal
    
    def _mppi_compute_weights(self, S):
        """
        MPPI Step 4: Compute trajectory weights from costs.
        
        w[i] = exp(-S[i] / λ) / Z, where Z = ∑ exp(-S[i] / λ)
        
        Args:
            S: (n_samples,) array of costs
        
        Returns:
            w: (n_samples,) array of normalized weights
        """
        # Numerical stability: subtract min cost
        S_norm = S - S.min()
        exp_S = np.exp(-S_norm / self.temperature)
        w = exp_S / exp_S.sum()
        return w
    
    def _mppi_weighted_control(self, U, w):
        """
        MPPI Step 5: Compute weighted mean of control sequences.
        
        u* = ∑ w[i] × U[i, 0]  (first control in each sequence)
        
        Args:
            U: (n_samples, n_steps, 2) control sequences
            w: (n_samples,) weights
        
        Returns:
            (v*, w*): optimal control to execute
        """
        u_star = np.sum(w[:, None] * U[:, 0, :], axis=0)
        return u_star[0], u_star[1]  # (v, w)
    


# ── Unit test / standalone demo ───────────────────────────────────────

if __name__ == "__main__":
    print("MppiCostmapPlanner unit test (mppi-costmap-v0, Phase 1 stub)")
    
    # Create instance
    le = MppiCostmapPlanner(n_samples=128)  # Smaller for unit test
    
    # Test API (non-blocking mailbox)
    assert not le.is_active(), "Should start inactive"
    
    le.set_goal(2.0, 1.0)
    assert le.is_active(), "Should be active after set_goal"
    
    # Dummy obs_map and pose
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    # Tick (geometric stub should return command now)
    cmd = le.tick(obs_map, pose, 0.033)
    print("  Geometric stub cmd:", cmd)
    assert cmd is not None, "Geometric stub should return command"
    assert 'fwd_mps' in cmd and 'ang_rads' in cmd, "Command should have fwd_mps, ang_rads"
    
    le.cancel()
    assert not le.is_active(), "Should be inactive after cancel"
    
    le.set_wander_mode(True)
    assert le.is_active(), "Should be active in wander mode"
    
    cmd = le.tick(obs_map, pose, 0.033)
    assert cmd is None, "Wander mode stub should return None (not implemented yet)"
    
    le.cancel()
    
    # Test debug state
    dbg = le.get_debug_state()
    assert 'goal_x' in dbg, "Debug state should have goal_x"
    
    # Test MPPI stubs (sampling, rollout, weighting)
    print("\n  Testing MPPI stubs...")
    u_prev = np.zeros((10, 2), dtype=np.float32)  # Match n_steps=10
    U = le._mppi_sample_trajectories(u_prev, n_samples=16, n_steps=10)
    assert U.shape == (16, 10, 2), "Sample shape mismatch"
    print("    ✓ _mppi_sample_trajectories")
    
    S, X = le._mppi_rollout_batch(U[:4], (0.0, 0.0, 0.0), obs_map)
    assert S.shape == (4,) and X.shape == (4, 10, 3), "Rollout shape mismatch"
    print("    ✓ _mppi_rollout_batch")
    
    w = le._mppi_compute_weights(S)
    assert abs(w.sum() - 1.0) < 1e-6, "Weights should sum to 1"
    print("    ✓ _mppi_compute_weights")
    
    v_star, w_star = le._mppi_weighted_control(U[:4], w)
    assert isinstance(v_star, (float, np.floating)), "Control should be float"
    print("    ✓ _mppi_weighted_control: v*=%.3f ω*=%.3f" % (v_star, w_star))
    
    print("\n✓ All unit tests passed (mppi-costmap-v0, Phase 1 stub)")
    print("  Next: Replace stubs with CUDA/Torch MPPI for full control")
