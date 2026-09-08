"""test_local_executive_keepouts.py - Verify keepouts paint_ego integration in local_executive.

This test verifies the fix for the NameError spam in Kevin's live logs:
- 'tools' module is properly imported
- tools.get_vision() is called (not the non-existent get_vision_instance())
"""

import ast
import sys


def test_tools_import():
    """Verify that tools module is imported in local_executive.py."""
    with open('src/local_executive.py') as f:
        tree = ast.parse(f.read())
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
    
    if 'tools' in imports:
        print("✓ 'tools' module is imported in local_executive.py")
        return True
    else:
        print("✗ 'tools' module is NOT imported in local_executive.py")
        return False


def test_get_vision_call():
    """Verify that tools.get_vision() is called (not get_vision_instance())."""
    with open('src/local_executive.py') as f:
        content = f.read()
    
    # Check that get_vision() is called
    if 'tools.get_vision()' in content:
        print("✓ tools.get_vision() is called correctly")
        has_correct_call = True
    else:
        print("✗ tools.get_vision() is NOT called")
        has_correct_call = False
    
    # Check that get_vision_instance() is NOT called
    if 'get_vision_instance()' not in content:
        print("✓ tools.get_vision_instance() is NOT called (good)")
        no_wrong_call = True
    else:
        print("✗ tools.get_vision_instance() is still called (BAD)")
        no_wrong_call = False
    
    return has_correct_call and no_wrong_call


def test_tools_module_has_get_vision():
    """Verify that tools.py has get_vision() function."""
    with open('src/tools.py') as f:
        tree = ast.parse(f.read())
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    
    if 'get_vision' in functions:
        print("✓ tools.py defines get_vision() function")
        return True
    else:
        print("✗ tools.py does NOT define get_vision() function")
        return False


def test_slam_locked_usage():
    """Verify that slam_locked is correctly extracted from vision instance."""
    with open('src/local_executive.py') as f:
        content = f.read()

    # Live path: default False, then set from vision when available.
    has_default = 'slam_locked = False' in content
    has_assign = 'slam_locked = vis.slam_locked' in content
    if has_default and has_assign:
        print("✓ slam_locked defaults False then set from vision")
        return True
    print("✗ slam_locked extraction pattern not found")
    return False


def main():
    print("=" * 60)
    print("Testing local_executive keepouts integration fix")
    print("=" * 60)
    print()
    
    test1 = test_tools_import()
    print()
    test2 = test_get_vision_call()
    print()
    test3 = test_tools_module_has_get_vision()
    print()
    test4 = test_slam_locked_usage()
    print()
    
    if test1 and test2 and test3 and test4:
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print()
        print("Fix summary:")
        print("- Added 'import tools' to local_executive.py")
        print("- Changed tools.get_vision_instance() → tools.get_vision()")
        print("- slam_locked is now correctly sourced from vision when available")
        print("- Falls back to False when vision is None")
        return 0
    else:
        print("=" * 60)
        print("❌ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
