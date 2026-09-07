"""
capture_timing.py - Per-stage timing for Vision._capture_loop profiling.

Tracks stage-by-stage timings to identify bottlenecks preventing 30 Hz capture.
Lightweight measurement overhead (<0.1ms per stage).

Usage:
    cap_timer = CaptureTimer(target_fps=30)
    
    with cap_timer.stage("grab"):
        # camera grab code
        pass
    
    # Every 90 frames:
    if cap_timer.frame_count % 90 == 0:
        stats = cap_timer.get_stats()
        cap_timer.print_report()
"""

import time
from typing import Dict, List, Optional
import numpy as np


class CaptureTimer:
    """
    Lightweight per-stage timing for capture loop profiling.
    
    Stages tracked:
    - grab: Camera parallel grabs (RS1, RS2, webcam)
    - topdown_depth: RS1 depth processing (depth_topdown)
    - topdown_checks: Near-field, overhang, soft-low, hazard checks
    - gpu_depth: RS2 GPU depth processing (scatter + morph)
    - obs_combine: Observation combining (blit + mask)
    - odom: Visual odometry (GPU or CPU)
    - gmap: Global map update
    - safety: Safety update
    - render: GPU atlas render
    
    Target: all stages combined < 33ms for 30 Hz
    """
    
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.target_ms = 1000.0 / target_fps
        
        self.frame_count = 0
        self.frame_start = 0.0
        self.frame_times: List[float] = []
        
        # Per-stage timing
        self.stage_times: Dict[str, List[float]] = {}
        self._current_stage: Optional[str] = None
        self._stage_start: float = 0.0
    
    def start_frame(self):
        """Call at the start of each capture loop iteration."""
        self.frame_start = time.monotonic()
    
    def end_frame(self):
        """Call at the end of each capture loop iteration."""
        frame_time = (time.monotonic() - self.frame_start) * 1000.0
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Keep last 300 frames
        if len(self.frame_times) > 300:
            self.frame_times = self.frame_times[-300:]
    
    def stage(self, name: str):
        """Context manager for timing a stage."""
        return _CaptureStageTimer(self, name)
    
    def record_stage(self, name: str, duration_ms: float):
        """Record a completed stage's timing."""
        if name not in self.stage_times:
            self.stage_times[name] = []
        self.stage_times[name].append(duration_ms)
        
        # Keep last 300 samples per stage
        if len(self.stage_times[name]) > 300:
            self.stage_times[name] = self.stage_times[name][-300:]
    
    def get_stats(self, window: int = 300) -> Dict:
        """Return timing statistics over last N frames."""
        if not self.frame_times:
            return {}
        
        recent_frames = self.frame_times[-window:]
        stats = {
            'frame_count': self.frame_count,
            'target_ms': self.target_ms,
            'frame_ms_mean': float(np.mean(recent_frames)),
            'frame_ms_p50': float(np.percentile(recent_frames, 50)),
            'frame_ms_p95': float(np.percentile(recent_frames, 95)),
            'frame_ms_max': float(np.max(recent_frames)),
            'fps_actual': 1000.0 / np.mean(recent_frames) if recent_frames else 0.0,
            'stages': {}
        }
        
        for stage, times in self.stage_times.items():
            recent = times[-window:]
            if recent:
                stats['stages'][stage] = {
                    'mean': float(np.mean(recent)),
                    'p50': float(np.percentile(recent, 50)),
                    'p95': float(np.percentile(recent, 95)),
                    'max': float(np.max(recent)),
                    'pct_budget': (np.mean(recent) / self.target_ms) * 100.0
                }
        
        return stats
    
    def print_report(self, window: int = 90):
        """Print timing report (call every N frames)."""
        if self.frame_count < 3:
            return
        
        stats = self.get_stats(window)
        if not stats:
            return
        
        print("=" * 80)
        print("CAPTURE TIMING REPORT (frames %d-%d)" % (
            self.frame_count - window + 1, self.frame_count))
        print("-" * 80)
        print("OVERALL: mean=%.1fms p50=%.1fms p95=%.1fms max=%.1fms | "
              "actual=%.1f Hz target=%.1f Hz" % (
                  stats['frame_ms_mean'], stats['frame_ms_p50'],
                  stats['frame_ms_p95'], stats['frame_ms_max'],
                  stats['fps_actual'], self.target_fps))
        print("-" * 80)
        
        # Sort stages by mean time (slowest first)
        stage_stats = stats.get('stages', {})
        sorted_stages = sorted(stage_stats.items(),
                              key=lambda x: x[1]['mean'],
                              reverse=True)
        
        print("%-18s %8s %8s %8s %8s %8s" % (
            "STAGE", "MEAN", "P50", "P95", "MAX", "%BUDGET"))
        print("-" * 80)
        
        total_mean = 0.0
        for stage, st in sorted_stages:
            total_mean += st['mean']
            print("%-18s %7.1fms %7.1fms %7.1fms %7.1fms %7.1f%%" % (
                stage, st['mean'], st['p50'], st['p95'], st['max'],
                st['pct_budget']))
        
        print("-" * 80)
        print("TOTAL STAGES: %.1fms (%.1f%% of target %.1fms)" % (
            total_mean, (total_mean / self.target_ms) * 100.0, self.target_ms))
        print("=" * 80)
    
    def get_bottlenecks(self, threshold_pct: float = 20.0) -> List[tuple]:
        """Return stages consuming >threshold% of frame budget.
        
        Returns list of (stage_name, mean_ms, pct_budget) sorted by time.
        """
        stats = self.get_stats()
        bottlenecks = []
        
        for stage, st in stats.get('stages', {}).items():
            if st['pct_budget'] > threshold_pct:
                bottlenecks.append((stage, st['mean'], st['pct_budget']))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks


class _CaptureStageTimer:
    """Context manager for timing a single capture stage."""
    
    def __init__(self, timer: CaptureTimer, name: str):
        self.timer = timer
        self.name = name
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_s = time.monotonic() - self.start_time
        duration_ms = duration_s * 1000.0
        self.timer.record_stage(self.name, duration_ms)
        return False
