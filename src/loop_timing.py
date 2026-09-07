"""
loop_timing.py - Per-stage timing and budget shedding for 30 Hz main loop.

Ensures critical path (atlas → safety → planner → wheels) never gets starved by
heavy droppable work (Rerun, get_frames, people/house_bot async calls, 3D recon).

Usage:
    shed = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
    
    with shed.stage("atlas"):
        atlas, ts = tools.get_atlas()
    
    if shed.should_run("rerun"):
        with shed.stage("rerun"):
            # expensive frame copies and logging
            ...
"""

import time
from typing import Optional, Dict


class FrameBudget:
    """
    Tracks per-stage timing and decides which droppable stages to skip
    when accumulated time exceeds a threshold.
    
    Priority levels:
    - CRITICAL (0): always run - atlas, safety, planner, wheel commands
    - DROPPABLE (1): skip if budget exceeded - rerun, get_frames, tool calls
    
    Stages:
        atlas       - get_atlas() call (critical)
        safety      - safety scale propagation (critical)
        planner     - gamepad + local_exec/navigator (critical)
        rerun       - RGB/depth/obs logging (droppable)
        tool_calls  - agent tool execution (droppable)
    """
    
    # Stage priorities
    CRITICAL = 0
    DROPPABLE = 1
    
    # Stage definitions: name -> priority level
    # Main loop stages:
    STAGE_PRIORITIES = {
        "atlas": CRITICAL,
        "safety": CRITICAL,
        "planner": CRITICAL,
        "rerun": DROPPABLE,
        "tool_calls": DROPPABLE,
    }
    
    # Vision capture loop stages:
    # See vision.py _capture_loop() for stage definitions
    CAPTURE_STAGE_PRIORITIES = {
        "grab": CRITICAL,           # Camera frame capture (blocking)
        "rs1_process": CRITICAL,    # Topdown depth (safety ground truth)
        "rs2_process": DROPPABLE,   # Forward depth (can fallback to topdown only)
        "obs_combine": CRITICAL,    # Combine obstacles (fast, needed for safety)
        "odom": DROPPABLE,          # Odometry update (throttleable with wheelbase=None)
        "gmap": DROPPABLE,          # Global map SLAM updates (expensive, non-safety-critical)
        "safety_update": CRITICAL,  # Safety scale computation (must run)
        "render": DROPPABLE,        # Atlas render (viz for humans/i777, not policy-critical)
    }
    
    def __init__(self, budget_ms: float, shed_threshold: float = 0.85, use_capture_priorities: bool = False):
        """
        Args:
            budget_ms: Frame time budget in milliseconds (33.3 for 30 Hz)
            shed_threshold: Fraction of budget (0.0-1.0) that triggers shedding
            use_capture_priorities: If True, use CAPTURE_STAGE_PRIORITIES instead of STAGE_PRIORITIES
        """
        self.budget_ms = budget_ms
        self.shed_threshold = shed_threshold
        self.shed_ms = budget_ms * shed_threshold
        self.use_capture_priorities = use_capture_priorities
        
        self.frame_start_time = 0.0
        self.accumulated_ms = 0.0
        self.stage_times: Dict[str, float] = {}
        self.stage_counts: Dict[str, int] = {}
        self.shed_counts: Dict[str, int] = {}
        
        self._stage_name: Optional[str] = None
        self._stage_start: float = 0.0
    
    def reset_frame(self):
        """Call at the start of each frame."""
        self.frame_start_time = time.monotonic()
        self.accumulated_ms = 0.0
        self.stage_times.clear()
    
    def elapsed_ms(self) -> float:
        """Time elapsed since frame start in milliseconds."""
        return (time.monotonic() - self.frame_start_time) * 1000.0
    
    def budget_exceeded(self) -> bool:
        """True if accumulated time exceeds shed threshold."""
        return self.accumulated_ms >= self.shed_ms
    
    def should_run(self, stage_name: str) -> bool:
        """
        Decide whether to run a stage.
        
        Critical stages always run.
        Droppable stages are skipped if budget is exceeded.
        """
        priorities = self.CAPTURE_STAGE_PRIORITIES if self.use_capture_priorities else self.STAGE_PRIORITIES
        priority = priorities.get(stage_name, self.DROPPABLE)
        
        if priority == self.CRITICAL:
            return True
        
        # Droppable stage - check budget
        if self.budget_exceeded():
            self.shed_counts[stage_name] = self.shed_counts.get(stage_name, 0) + 1
            return False
        
        return True
    
    def stage(self, name: str):
        """
        Context manager for timing a stage.
        
        Usage:
            with shed.stage("atlas"):
                atlas, ts = tools.get_atlas()
        """
        return _StageTimer(self, name)
    
    def record_stage(self, name: str, duration_ms: float):
        """Record a completed stage's timing."""
        self.stage_times[name] = duration_ms
        self.accumulated_ms += duration_ms
        self.stage_counts[name] = self.stage_counts.get(name, 0) + 1
    
    def get_stats(self) -> Dict[str, float]:
        """Return current frame timing statistics."""
        return {
            "elapsed_ms": self.elapsed_ms(),
            "accumulated_ms": self.accumulated_ms,
            "budget_ms": self.budget_ms,
            "budget_used_pct": (self.accumulated_ms / self.budget_ms) * 100.0,
            "stage_times": dict(self.stage_times),
        }
    
    def get_lifetime_stats(self) -> Dict:
        """Return lifetime statistics for all stages."""
        return {
            "stage_counts": dict(self.stage_counts),
            "shed_counts": dict(self.shed_counts),
        }


class _StageTimer:
    """Context manager for timing a single stage."""
    
    def __init__(self, budget: FrameBudget, name: str):
        self.budget = budget
        self.name = name
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_s = time.monotonic() - self.start_time
        duration_ms = duration_s * 1000.0
        self.budget.record_stage(self.name, duration_ms)
        return False
