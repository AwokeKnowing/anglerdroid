"""Unsafe policy example - demonstrates collision detection.

This policy intentionally overrides safety stops to prove the hypothesis 
that the couch crash was caused by recover/commit forcing motion 
when fwd_scale=0.
"""

from sim.policy import Policy


class UnsafeCommitPolicy(Policy):
    """UNSAFE: Forces forward motion even when fwd_scale=0.
    
    This mimics the suspected behavior that caused the real couch crash.
    Use for testing collision detection only!
    """
    
    def __init__(self):
        self.stuck_counter = 0
        self.commit_mode = False
    
    def reset(self):
        self.stuck_counter = 0
        self.commit_mode = False
    
    def act(self, obs, height, safety_scales, pose):
        fwd_scale = safety_scales['fwd']
        
        if fwd_scale < 0.3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.commit_mode = False
        
        if self.stuck_counter > 10:
            self.commit_mode = True
        
        if self.commit_mode:
            return 0.15, 0.0
        elif fwd_scale > 0.5:
            return 0.15, 0.0
        else:
            return 0.0, 0.0
