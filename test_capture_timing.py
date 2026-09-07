"""
test_capture_timing.py - Unit tests for capture timing instrumentation.

Tests the CaptureTimer module without requiring camera hardware.
"""

import time
import numpy as np
from src.capture_timing import CaptureTimer


def test_capture_timer_basic():
    """Test basic CaptureTimer functionality."""
    print("test_capture_timer_basic: start")
    
    timer = CaptureTimer(target_fps=30.0)
    assert timer.target_ms == 1000.0 / 30.0
    assert timer.frame_count == 0
    
    # Simulate a frame
    timer.start_frame()
    
    with timer.stage("test_stage"):
        time.sleep(0.001)  # 1ms
    
    timer.end_frame()
    
    assert timer.frame_count == 1
    assert "test_stage" in timer.stage_times
    assert len(timer.stage_times["test_stage"]) == 1
    
    print("test_capture_timer_basic: PASS")


def test_capture_timer_stats():
    """Test timing statistics computation."""
    print("test_capture_timer_stats: start")
    
    timer = CaptureTimer(target_fps=30.0)
    
    # Simulate 10 frames with known timings
    for _ in range(10):
        timer.start_frame()
        
        with timer.stage("fast"):
            time.sleep(0.001)  # ~1ms
        
        with timer.stage("slow"):
            time.sleep(0.005)  # ~5ms
        
        timer.end_frame()
    
    stats = timer.get_stats(window=10)
    
    assert stats['frame_count'] == 10
    assert 'fast' in stats['stages']
    assert 'slow' in stats['stages']
    
    # Slow stage should take more time than fast stage
    assert stats['stages']['slow']['mean'] > stats['stages']['fast']['mean']
    
    print("test_capture_timer_stats: PASS")
    print("  fast stage: %.2fms" % stats['stages']['fast']['mean'])
    print("  slow stage: %.2fms" % stats['stages']['slow']['mean'])


def test_capture_timer_bottlenecks():
    """Test bottleneck detection."""
    print("test_capture_timer_bottlenecks: start")
    
    timer = CaptureTimer(target_fps=30.0)  # 33.3ms budget
    
    # Simulate frames with one heavy stage
    for _ in range(5):
        timer.start_frame()
        
        with timer.stage("light"):
            time.sleep(0.001)  # 1ms (~3% of budget)
        
        with timer.stage("heavy"):
            time.sleep(0.010)  # 10ms (~30% of budget)
        
        timer.end_frame()
    
    # Get bottlenecks (>20% threshold)
    bottlenecks = timer.get_bottlenecks(threshold_pct=20.0)
    
    assert len(bottlenecks) == 1
    assert bottlenecks[0][0] == "heavy"
    assert bottlenecks[0][2] > 20.0  # pct_budget
    
    print("test_capture_timer_bottlenecks: PASS")
    print("  bottleneck detected: %s at %.1f%% of budget" % 
          (bottlenecks[0][0], bottlenecks[0][2]))


def test_capture_timer_percentiles():
    """Test percentile statistics (p50, p95)."""
    print("test_capture_timer_percentiles: start")
    
    timer = CaptureTimer(target_fps=30.0)
    
    # Simulate frames with varying timings
    sleep_times = [0.001, 0.002, 0.003, 0.010, 0.002, 0.003, 0.002, 0.020, 0.003, 0.002]
    
    for sleep_time in sleep_times:
        timer.start_frame()
        
        with timer.stage("variable"):
            time.sleep(sleep_time)
        
        timer.end_frame()
    
    stats = timer.get_stats(window=10)
    
    assert 'variable' in stats['stages']
    stage = stats['stages']['variable']
    
    # p95 should be higher than p50
    assert stage['p95'] > stage['p50']
    
    # p95 should be close to the outliers (10ms, 20ms)
    assert stage['p95'] > 8.0  # Should capture the 10ms+ outliers
    
    print("test_capture_timer_percentiles: PASS")
    print("  p50=%.2fms p95=%.2fms max=%.2fms" % 
          (stage['p50'], stage['p95'], stage['max']))


def test_capture_timer_window():
    """Test sliding window behavior."""
    print("test_capture_timer_window: start")
    
    timer = CaptureTimer(target_fps=30.0)
    
    # Simulate 400 frames (more than the 300-frame window)
    for i in range(400):
        timer.start_frame()
        
        with timer.stage("test"):
            time.sleep(0.0001)  # Very fast
        
        timer.end_frame()
    
    # Check that window is maintained at 300
    assert len(timer.frame_times) == 300
    assert len(timer.stage_times["test"]) == 300
    
    print("test_capture_timer_window: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CaptureTimer module")
    print("=" * 60)
    
    test_capture_timer_basic()
    test_capture_timer_stats()
    test_capture_timer_bottlenecks()
    test_capture_timer_percentiles()
    test_capture_timer_window()
    
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)
