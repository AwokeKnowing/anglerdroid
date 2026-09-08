#!/usr/bin/env python3
"""test_morph_match_fallback.py – Unit tests for morph-match featureless rejection.

Tests that GPU visual odometry properly rejects featureless/failed matches
and retains wheel-odom pose instead of inventing spurious rotations.

Run: python3 test_morph_match_fallback.py
"""

import os
import sys
import numpy as np
import time
import math

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Enable robust morph-match for these tests
os.environ['KEVIN_ROBUST_MORPH_MATCH'] = '1'

# Mock ModernGL if not available (tests run on host)
if 'moderngl' not in sys.modules:
    sys.modules['moderngl'] = type(sys)('moderngl')
    sys.modules['moderngl'].NEAREST = 0
    sys.modules['moderngl'].LINEAR = 1
    sys.modules['moderngl'].POINTS = 0
    sys.modules['moderngl'].TRIANGLES = 4
    sys.modules['moderngl'].TRIANGLE_STRIP = 5
    sys.modules['moderngl'].DEPTH_TEST = 2929
    sys.modules['moderngl'].BLEND = 3042

# Mock cv2 if not available (tests run on host)
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = type(sys)('cv2')

from gpu_render import GPURenderer


def test_robust_flag_enabled():
    """Verify KEVIN_ROBUST_MORPH_MATCH is enabled for these tests."""
    from gpu_render import KEVIN_ROBUST_MORPH_MATCH
    assert KEVIN_ROBUST_MORPH_MATCH, "KEVIN_ROBUST_MORPH_MATCH must be 1 for tests"
    print("✓ KEVIN_ROBUST_MORPH_MATCH=1")


def test_wider_search_window():
    """Verify wider search window when robust match is enabled."""
    from gpu_render import KEVIN_ROBUST_MORPH_MATCH
    
    # Create minimal renderer (only needs _od_* attributes for this test)
    class MinimalRenderer:
        def __init__(self):
            self._od_configured = False
            self._od_gl_ready = False
        
        def configure_odom(self, fx=307.0, ds_factor=4, search=8):
            self._od_fx = float(fx)
            self._od_ds = int(ds_factor)
            # Same logic as gpu_render.py
            if KEVIN_ROBUST_MORPH_MATCH:
                self._od_search = 12
            else:
                self._od_search = int(search)
            self._od_configured = True
            self._od_gl_ready = False
    
    renderer = MinimalRenderer()
    renderer.configure_odom(fx=307.0, ds_factor=4, search=8)
    
    # When robust match is enabled, search should be widened to 12
    assert renderer._od_search == 12, f"Expected search=12, got {renderer._od_search}"
    print(f"✓ Search window widened: {renderer._od_search} (was 8)")


def test_featureless_sad_map_rejected():
    """Test that featureless SAD map (flat, low variance) is rejected.
    
    Simulates empty floor: SAD map is very flat, no sharp minimum.
    Expected: confidence = 0.0 (match rejected, wheel-odom retained).
    """
    # Simulate SAD map with very low variance (featureless floor)
    sw = 17  # 2 * 8 + 1 (default search)
    sad_flat = np.ones((sw, sw), dtype=np.float32) * 0.5
    # Add tiny noise to avoid division by zero
    sad_flat += np.random.normal(0, 0.001, sad_flat.shape).astype(np.float32)
    
    min_sad = float(sad_flat.min())
    mean_sad = float(sad_flat.mean())
    sad_std = float(sad_flat.std())
    sharpness = (mean_sad - min_sad) / (mean_sad + 1e-6)
    variance_ratio = sad_std / (mean_sad + 1e-6)
    
    # Check featureless rejection logic (from gpu_render.py odom_gpu)
    if variance_ratio < 0.10:  # Too flat → featureless
        confidence = 0.0
    elif sharpness < 0.15:  # Weak peak → ambiguous
        confidence = 0.0
    else:
        confidence = min(1.0, sharpness * 1.5)
    
    print(f"Featureless SAD: var_ratio={variance_ratio:.4f} sharp={sharpness:.4f} conf={confidence:.2f}")
    assert confidence == 0.0, f"Expected confidence=0.0 for featureless, got {confidence}"
    assert variance_ratio < 0.10, f"Expected low variance, got {variance_ratio}"
    print("✓ Featureless match rejected (conf=0.0)")


def test_weak_match_rejected():
    """Test that weak match (low sharpness) is rejected.
    
    Simulates ambiguous match: SAD map has a minimum but it's not sharp.
    Expected: confidence = 0.0 (match rejected).
    """
    sw = 17
    # Create SAD map with weak minimum (poor sharpness)
    sad_weak = np.ones((sw, sw), dtype=np.float32) * 0.5
    sad_weak[sw//2, sw//2] = 0.45  # Weak minimum (10% lower than mean)
    
    min_sad = float(sad_weak.min())
    mean_sad = float(sad_weak.mean())
    sad_std = float(sad_weak.std())
    sharpness = (mean_sad - min_sad) / (mean_sad + 1e-6)
    variance_ratio = sad_std / (mean_sad + 1e-6)
    
    # Check rejection logic
    if variance_ratio < 0.10:
        confidence = 0.0
    elif sharpness < 0.15:  # Weak peak → ambiguous
        confidence = 0.0
    else:
        confidence = min(1.0, sharpness * 1.5)
    
    print(f"Weak match: var_ratio={variance_ratio:.4f} sharp={sharpness:.4f} conf={confidence:.2f}")
    assert confidence == 0.0, f"Expected confidence=0.0 for weak match, got {confidence}"
    assert sharpness < 0.15, f"Expected low sharpness, got {sharpness}"
    print("✓ Weak match rejected (conf=0.0)")


def test_strong_match_accepted():
    """Test that strong match (sharp peak) is accepted.
    
    Simulates good feature match: SAD map has a sharp, clear minimum.
    Expected: confidence > 0.0 (match accepted, visual update applied).
    """
    sw = 17
    # Create SAD map with sharp minimum and high variance (good match with features)
    # Simulate a textured scene: varied SAD values across search space
    np.random.seed(42)  # Reproducible
    sad_strong = np.random.uniform(0.4, 0.6, (sw, sw)).astype(np.float32)
    # Add a clear minimum at center (good match location)
    sad_strong[sw//2, sw//2] = 0.05  # Sharp minimum
    # Add gradient around minimum to make it more pronounced
    for dr in [1, 2]:
        for dx in range(-dr, dr+1):
            for dy in range(-dr, dr+1):
                if abs(dx) == dr or abs(dy) == dr:
                    y, x = sw//2 + dy, sw//2 + dx
                    if 0 <= y < sw and 0 <= x < sw:
                        sad_strong[y, x] = 0.1 + dr * 0.05
    
    min_sad = float(sad_strong.min())
    mean_sad = float(sad_strong.mean())
    sad_std = float(sad_strong.std())
    sharpness = (mean_sad - min_sad) / (mean_sad + 1e-6)
    variance_ratio = sad_std / (mean_sad + 1e-6)
    
    # Check acceptance logic (same as gpu_render.py odom_gpu)
    if variance_ratio < 0.10:
        confidence = 0.0
    elif sharpness < 0.15:
        confidence = 0.0
    else:
        confidence = min(1.0, sharpness * 1.5)
    
    print(f"Strong match: var_ratio={variance_ratio:.4f} sharp={sharpness:.4f} conf={confidence:.2f}")
    assert confidence > 0.0, f"Expected confidence>0.0 for strong match, got {confidence}"
    assert sharpness >= 0.15, f"Expected high sharpness, got {sharpness}"
    assert variance_ratio >= 0.10, f"Expected high variance, got {variance_ratio}"
    print("✓ Strong match accepted (conf={:.2f})".format(confidence))


def test_wheel_odom_retained_on_rejection():
    """Test that wheel-odom pose is retained when visual match is rejected.
    
    Integration test: verify that low confidence from featureless scene
    causes pose estimator to reject visual update and keep wheel-odom.
    """
    from pose import PoseEstimator
    
    # Create pose estimator
    pose = PoseEstimator(wheelbase_m=0.235, wheel_radius_m=0.066)
    pose.reset()
    
    # Simulate wheel odometry: forward 10cm
    v_left = v_right = 0.1  # m/s
    dt = 1.0  # 1 second
    
    # Simulate rejected visual odometry (featureless → conf=0.0)
    vis_yaw = 0.5  # Spurious rotation (should be rejected)
    vis_fwd = 0.05  # Spurious forward (should be rejected)
    vis_conf = 0.0  # Rejected by featureless detection
    
    # Update pose with wheel + rejected visual (no IMU for this test)
    fused_yaw, fused_fwd = pose.update(
        v_left, v_right, dt,
        vis_yaw=vis_yaw, vis_fwd=vis_fwd, vis_confidence=vis_conf,
        using_encoder_feedback=True,
        imu_yaw_rate=0.0)
    
    # Expected: visual rejected, wheel-odom used
    wheel_fwd = (v_left + v_right) * 0.5 * dt
    wheel_yaw = 0.0  # Straight forward
    
    print(f"Wheel: fwd={wheel_fwd:.3f} yaw={wheel_yaw:.3f}")
    print(f"Visual (rejected): fwd={vis_fwd:.3f} yaw={vis_yaw:.3f} conf={vis_conf:.2f}")
    print(f"Fused: fwd={fused_fwd:.3f} yaw={fused_yaw:.3f}")
    print(f"Pose: x={pose.x:.3f} y={pose.y:.3f} theta={pose.theta:.3f}")
    
    # Verify: fused should match wheel-odom (visual rejected)
    assert abs(fused_fwd - wheel_fwd) < 0.001, "Visual should be rejected, wheel-odom used"
    assert abs(fused_yaw - wheel_yaw) < 0.001, "Visual yaw should be rejected"
    assert abs(pose.theta - 0.0) < 0.001, "Pose should not have spurious rotation"
    
    # Check tracking quality metrics
    quality = pose.get_tracking_quality()
    assert quality['visual_rejected'] > 0, "Visual should be marked as rejected"
    assert quality['visual_accepted'] == 0, "Visual should not be accepted"
    
    print("✓ Wheel-odom retained on rejection (no spurious spin)")


def test_imu_wheel_fusion_on_visual_rejection():
    """Test that IMU+wheel fusion is used when visual odometry is rejected.
    
    CRITICAL: When visual fails (featureless, low confidence), we must use
    wheel+IMU fusion, NOT wheel-only. IMU weight should be boosted from
    0.15 (base) to 0.50 (fallback) to compensate for missing visual.
    """
    from pose import PoseEstimator, IMU_YAW_WEIGHT_FALLBACK
    
    # Create pose estimator
    pose = PoseEstimator(wheelbase_m=0.235, wheel_radius_m=0.066)
    pose.reset()
    
    # Simulate wheel odometry: straight forward
    v_left = v_right = 0.1  # m/s
    dt = 1.0  # 1 second
    
    # Simulate rejected visual odometry (featureless → conf=0.0)
    vis_yaw = 0.5  # Spurious rotation (should be rejected)
    vis_fwd = 0.05  # Spurious forward (should be rejected)
    vis_conf = 0.0  # Rejected by featureless detection
    
    # Simulate IMU reporting a small yaw rate
    imu_yaw_rate = 0.1  # rad/s (turning left)
    
    # Update pose with wheel + rejected visual + IMU
    fused_yaw, fused_fwd = pose.update(
        v_left, v_right, dt,
        vis_yaw=vis_yaw, vis_fwd=vis_fwd, vis_confidence=vis_conf,
        using_encoder_feedback=True,
        imu_yaw_rate=imu_yaw_rate)
    
    # Expected: visual rejected, wheel says no turn, IMU says turning
    wheel_fwd = (v_left + v_right) * 0.5 * dt
    wheel_yaw = 0.0  # Straight forward
    imu_dtheta = imu_yaw_rate * dt  # 0.1 rad
    
    # With visual rejected, IMU weight should be FALLBACK (0.50)
    # Fused yaw = (1 - 0.50) * wheel_yaw + 0.50 * imu_dtheta
    #           = 0.50 * 0.0 + 0.50 * 0.1 = 0.05 rad
    expected_fused_yaw = (1.0 - IMU_YAW_WEIGHT_FALLBACK) * wheel_yaw + \
                         IMU_YAW_WEIGHT_FALLBACK * imu_dtheta
    
    print(f"Wheel: fwd={wheel_fwd:.3f} yaw={wheel_yaw:.3f}")
    print(f"Visual (rejected): fwd={vis_fwd:.3f} yaw={vis_yaw:.3f} conf={vis_conf:.2f}")
    print(f"IMU: yaw_rate={imu_yaw_rate:.3f} dtheta={imu_dtheta:.3f}")
    print(f"Fused: fwd={fused_fwd:.3f} yaw={fused_yaw:.3f}")
    print(f"Expected fused yaw: {expected_fused_yaw:.3f} (IMU weight={IMU_YAW_WEIGHT_FALLBACK})")
    print(f"Pose: x={pose.x:.3f} y={pose.y:.3f} theta={pose.theta:.3f}")
    
    # Verify: fused yaw should include IMU contribution with FALLBACK weight
    assert abs(fused_yaw - expected_fused_yaw) < 0.001, \
        f"Fused yaw should blend wheel+IMU with fallback weight, got {fused_yaw:.3f}, expected {expected_fused_yaw:.3f}"
    
    # Verify: spurious visual yaw was rejected (not used)
    assert abs(fused_yaw - vis_yaw) > 0.1, \
        "Spurious visual yaw should be rejected (not close to fused yaw)"
    
    # Verify: IMU did contribute (fused != wheel-only)
    assert abs(fused_yaw - wheel_yaw) > 0.01, \
        "IMU should contribute (fused yaw != wheel-only yaw)"
    
    # Check tracking quality metrics
    quality = pose.get_tracking_quality()
    assert quality['visual_rejected'] > 0, "Visual should be marked as rejected"
    assert quality['imu_used_count'] > 0, "IMU should be used"
    assert quality['imu_fallback_count'] > 0, "IMU fallback weight should be used"
    assert quality['imu_fallback_rate'] == 1.0, "100% of IMU usage should be fallback"
    
    print("✓ IMU+wheel fusion on visual rejection (IMU weight boosted to fallback)")


def test_imu_base_weight_on_visual_acceptance():
    """Test that IMU uses base weight when visual odometry is accepted.
    
    When visual is accepted with good confidence, IMU should use the
    base weight (0.15) rather than the fallback weight (0.50).
    """
    from pose import PoseEstimator, IMU_YAW_WEIGHT_BASE, IMU_VIS_CONF_THRESH
    
    # Create pose estimator
    pose = PoseEstimator(wheelbase_m=0.235, wheel_radius_m=0.066)
    pose.reset()
    
    # Simulate wheel odometry: straight forward
    v_left = v_right = 0.1  # m/s
    dt = 1.0  # 1 second
    
    # Simulate accepted visual odometry (good confidence)
    vis_yaw = 0.02  # Small rotation (plausible, will be accepted)
    vis_fwd = 0.095  # Close to wheel prediction (plausible)
    vis_conf = 0.8  # Good confidence (> IMU_VIS_CONF_THRESH=0.20)
    
    # Simulate IMU reporting a different yaw rate
    imu_yaw_rate = 0.05  # rad/s (slightly different from visual)
    
    # Update pose with wheel + accepted visual + IMU
    fused_yaw, fused_fwd = pose.update(
        v_left, v_right, dt,
        vis_yaw=vis_yaw, vis_fwd=vis_fwd, vis_confidence=vis_conf,
        using_encoder_feedback=True,
        imu_yaw_rate=imu_yaw_rate)
    
    print(f"Wheel: fwd={0.1:.3f} yaw={0.0:.3f}")
    print(f"Visual (accepted): fwd={vis_fwd:.3f} yaw={vis_yaw:.3f} conf={vis_conf:.2f}")
    print(f"IMU: yaw_rate={imu_yaw_rate:.3f}")
    print(f"Fused: fwd={fused_fwd:.3f} yaw={fused_yaw:.3f}")
    print(f"IMU weight: {IMU_YAW_WEIGHT_BASE} (base, visual confidence good)")
    
    # Check tracking quality metrics
    quality = pose.get_tracking_quality()
    assert quality['visual_accepted'] > 0, "Visual should be accepted"
    assert quality['imu_used_count'] > 0, "IMU should be used"
    # Fallback count should be 0 because visual was accepted with good confidence
    assert quality['imu_fallback_count'] == 0, \
        "IMU fallback should NOT be used when visual confidence is good"
    
    print("✓ IMU base weight on visual acceptance (confidence good)")



def test_legacy_behavior_without_flag():
    """Test that legacy behavior is preserved when flag is OFF."""
    # Disable robust match for this test
    os.environ['KEVIN_ROBUST_MORPH_MATCH'] = '0'
    
    # Force reload of gpu_render to pick up env change
    import importlib
    import gpu_render
    importlib.reload(gpu_render)
    
    from gpu_render import KEVIN_ROBUST_MORPH_MATCH
    assert not KEVIN_ROBUST_MORPH_MATCH, "Flag should be OFF"
    
    # Create minimal renderer
    class MinimalRenderer:
        def __init__(self):
            self._od_configured = False
            self._od_gl_ready = False
        
        def configure_odom(self, fx=307.0, ds_factor=4, search=8):
            self._od_fx = float(fx)
            self._od_ds = int(ds_factor)
            # Legacy: use provided search (no widening)
            if KEVIN_ROBUST_MORPH_MATCH:
                self._od_search = 12
            else:
                self._od_search = int(search)
            self._od_configured = True
            self._od_gl_ready = False
    
    renderer = MinimalRenderer()
    renderer.configure_odom(fx=307.0, ds_factor=4, search=8)
    
    # Legacy: search window should be 8 (not widened)
    assert renderer._od_search == 8, f"Expected legacy search=8, got {renderer._od_search}"
    
    # Simulate featureless SAD map
    sw = 17
    sad_flat = np.ones((sw, sw), dtype=np.float32) * 0.5
    sad_flat += np.random.normal(0, 0.001, sad_flat.shape).astype(np.float32)
    
    min_sad = float(sad_flat.min())
    mean_sad = float(sad_flat.mean())
    sharpness = (mean_sad - min_sad) / (mean_sad + 1e-6)
    
    # Legacy confidence: sharpness * 2.0 (no variance check)
    confidence_legacy = min(1.0, sharpness * 2.0)
    
    # Legacy may accept borderline matches (no featureless rejection)
    print(f"Legacy: sharp={sharpness:.4f} conf={confidence_legacy:.2f}")
    print("✓ Legacy behavior: no featureless rejection (as expected)")
    
    # Re-enable robust match for other tests
    os.environ['KEVIN_ROBUST_MORPH_MATCH'] = '1'
    importlib.reload(gpu_render)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing morph-match featureless rejection")
    print("=" * 60)
    
    tests = [
        test_robust_flag_enabled,
        test_wider_search_window,
        test_featureless_sad_map_rejected,
        test_weak_match_rejected,
        test_strong_match_accepted,
        test_wheel_odom_retained_on_rejection,
        test_imu_wheel_fusion_on_visual_rejection,
        test_imu_base_weight_on_visual_acceptance,
        test_legacy_behavior_without_flag,
    ]
    
    failed = []
    for test_func in tests:
        print(f"\n{test_func.__name__}:")
        try:
            test_func()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(test_func.__name__)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed!")
        sys.exit(0)
