"""Pluggable policy interface + HouseBotLite reference implementation.

HouseBotLite ports the scoring/early-divert/back→spin→commit idea 
BUT refuses to override hard stops from SafetyGuard.
"""

import math
import numpy as np


class Policy:
    """Base policy interface."""
    
    def reset(self):
        """Reset policy state at start of episode."""
        pass
    
    def act(self, obs, height, safety_scales, pose):
        """Compute action from observation.
        
        Args:
            obs: (H, W) uint8 ego obstacle map (0=clear, 255=occupied)
            height: (H, W) uint8 ego height map (cm above floor)
            safety_scales: dict with 'fwd', 'bwd', 'ang' scales (0-1)
            pose: dict with 'x', 'y', 'theta' (world coords)
        
        Returns:
            (v, w) - linear and angular velocity commands (m/s, rad/s)
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    """Random walk for testing."""
    
    def __init__(self, v_max=0.1, w_max=0.5):
        self.v_max = v_max
        self.w_max = w_max
    
    def act(self, obs, height, safety_scales, pose):
        v = np.random.uniform(-self.v_max, self.v_max)
        w = np.random.uniform(-self.w_max, self.w_max)
        return v, w


class HouseBotLite(Policy):
    """Reference policy: look-before-leap with recover fallback.
    
    Key principle: NEVER override safety hard stops (scale=0).
    When stuck (fwd_scale=0), try:
      1. Back up (if bwd_scale > 0)
      2. Spin (if ang_scale > 0)
      3. Commit forward ONLY if fwd_scale rises above threshold
    
    This prevents the couch crash where recover/commit drove through obstacles.
    """
    
    def __init__(self, v_cruise=0.15, w_cruise=0.4):
        self.v_cruise = v_cruise
        self.w_cruise = w_cruise
        self.mode = "cruise"
        self.stuck_counter = 0
        self.backup_counter = 0
        self.spin_counter = 0
        self.backup_duration = 20
        self.spin_duration = 30
    
    def reset(self):
        self.mode = "cruise"
        self.stuck_counter = 0
        self.backup_counter = 0
        self.spin_counter = 0
    
    def act(self, obs, height, safety_scales, pose):
        fwd_scale = safety_scales['fwd']
        bwd_scale = safety_scales['bwd']
        ang_scale = safety_scales['ang']
        
        if self.mode == "cruise":
            if fwd_scale > 0.5:
                return self.v_cruise, 0.0
            else:
                self.stuck_counter += 1
                if self.stuck_counter > 5:
                    self.mode = "backup"
                    self.backup_counter = 0
                    self.stuck_counter = 0
                return 0.0, 0.0
        
        elif self.mode == "backup":
            self.backup_counter += 1
            
            if bwd_scale <= 0:
                self.mode = "spin"
                self.spin_counter = 0
                return 0.0, 0.0
            
            if self.backup_counter >= self.backup_duration:
                self.mode = "spin"
                self.spin_counter = 0
                return 0.0, 0.0
            
            return -0.1, 0.0
        
        elif self.mode == "spin":
            self.spin_counter += 1
            
            if ang_scale <= 0:
                if bwd_scale > 0:
                    self.mode = "backup"
                    self.backup_counter = 0
                    return 0.0, 0.0
                else:
                    return 0.0, 0.0
            
            if fwd_scale > 0.8:
                self.mode = "cruise"
                self.stuck_counter = 0
                return 0.0, 0.0
            
            if self.spin_counter >= self.spin_duration:
                if fwd_scale > 0.3:
                    self.mode = "cruise"
                    self.stuck_counter = 0
                    return 0.0, 0.0
                else:
                    self.mode = "backup"
                    self.backup_counter = 0
                    return 0.0, 0.0
            
            return 0.0, self.w_cruise
        
        return 0.0, 0.0


class StopPolicy(Policy):
    """Always output zero velocity."""
    
    def act(self, obs, height, safety_scales, pose):
        return 0.0, 0.0


def create_policy(name: str):
    """Factory for creating policies by name."""
    if name == "random":
        return RandomPolicy()
    elif name == "housebot":
        return HouseBotLite()
    elif name == "stop":
        return StopPolicy()
    else:
        raise ValueError(f"Unknown policy: {name}")
