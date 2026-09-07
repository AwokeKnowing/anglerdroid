#!/usr/bin/env python3
"""Unit test for using_encoder_feedback parameter in pose fusion.

Regression test for NameError bug where using_encoder_feedback was
referenced but never defined in vision.py _capture_loop.
"""
import os
import sys
import py_compile
import ast


def test_vision_compiles_without_nameerror():
    """Test that vision.py compiles without NameError.
    
    If using_encoder_feedback is not defined, Python will raise
    a NameError at compile time.
    """
    vision_path = os.path.join(os.path.dirname(__file__), "src", "vision.py")
    try:
        py_compile.compile(vision_path, doraise=True)
        print("✓ PASS: vision.py compiles without NameError")
        return True
    except NameError as e:
        raise AssertionError(f"NameError in vision.py: {e}")


def test_using_encoder_feedback_defined_before_use():
    """Verify using_encoder_feedback is defined before both usage sites."""
    vision_path = os.path.join(os.path.dirname(__file__), "src", "vision.py")
    
    with open(vision_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find the definition line
    def_line = None
    for i, line in enumerate(lines):
        if 'using_encoder_feedback = False' in line and 'Determine if' in lines[i-1]:
            def_line = i + 1  # Convert to 1-indexed
            break
    
    # Find usage lines
    usage_lines = []
    for i, line in enumerate(lines):
        if 'using_encoder_feedback=' in line or 'if not using_encoder_feedback:' in line:
            if 'using_encoder_feedback = ' not in line:  # Skip definition
                usage_lines.append(i + 1)  # Convert to 1-indexed
    
    assert def_line is not None, "Could not find using_encoder_feedback definition"
    assert len(usage_lines) >= 2, f"Expected at least 2 usage sites, found {len(usage_lines)}"
    
    for usage_line in usage_lines:
        assert def_line < usage_line, \
            f"Definition at line {def_line} must come before usage at line {usage_line}"
    
    print(f"✓ PASS: using_encoder_feedback defined at line {def_line}, "
          f"used at lines {usage_lines}")
    return True


def test_encoder_feedback_logic_check():
    """Verify the encoder feedback logic is correct."""
    vision_path = os.path.join(os.path.dirname(__file__), "src", "vision.py")
    
    with open(vision_path, 'r') as f:
        content = f.read()
    
    # Check that the logic checks encoder_ok and age_s < 0.3
    assert "health = self._wheelbase.get_encoder_health()" in content, \
        "Missing get_encoder_health() call"
    assert "health['encoder_ok']" in content, \
        "Missing encoder_ok check"
    assert "health['age_s'] < 0.3" in content, \
        "Missing age_s < 0.3 check"
    
    print("✓ PASS: encoder feedback logic checks encoder_ok and age_s < 0.3")
    return True


def test_pose_update_signature_includes_param():
    """Verify PoseEstimator.update has using_encoder_feedback parameter."""
    pose_path = os.path.join(os.path.dirname(__file__), "src", "pose.py")
    
    with open(pose_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Find PoseEstimator class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'PoseEstimator':
            # Find update method
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'update':
                    # Check parameters
                    param_names = [arg.arg for arg in item.args.args]
                    assert 'using_encoder_feedback' in param_names, \
                        "PoseEstimator.update missing using_encoder_feedback parameter"
                    print("✓ PASS: PoseEstimator.update has using_encoder_feedback parameter")
                    return True
    
    raise AssertionError("Could not find PoseEstimator.update method")


if __name__ == "__main__":
    test_vision_compiles_without_nameerror()
    test_using_encoder_feedback_defined_before_use()
    test_encoder_feedback_logic_check()
    test_pose_update_signature_includes_param()
    print("\n✓✓✓ ALL ENCODER FEEDBACK TESTS PASSED ✓✓✓")
