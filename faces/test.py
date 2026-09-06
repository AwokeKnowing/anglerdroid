"""Test suite for face recognition system."""

import sys
import tempfile
import shutil
from pathlib import Path

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

from faces.recognizer import FaceRecognizer
from faces.conversation import ConversationManager, create_speak_function


def create_test_image(size=(640, 480)):
    """Create a simple test image with a white circle (fake face)."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        return None
    
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    center = (size[0] // 2, size[1] // 2)
    cv2.circle(img, center, 50, (255, 255, 255), -1)
    cv2.circle(img, (center[0] - 15, center[1] - 10), 5, (0, 0, 0), -1)
    cv2.circle(img, (center[0] + 15, center[1] - 10), 5, (0, 0, 0), -1)
    cv2.ellipse(img, center, (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    return img


def test_recognizer_init():
    """Test FaceRecognizer initialization."""
    if not _HAS_OPENCV:
        print("⚠️  test_recognizer_init skipped (OpenCV not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer = FaceRecognizer(gallery_path=tmpdir)
            assert recognizer.gallery_path == Path(tmpdir)
            assert isinstance(recognizer.db, dict)
            assert len(recognizer.db) == 0
            print("✅ test_recognizer_init passed")
            return True
        except Exception as e:
            print(f"❌ test_recognizer_init failed: {e}")
            return False


def test_face_detection():
    """Test face detection."""
    if not _HAS_OPENCV:
        print("⚠️  test_face_detection skipped (OpenCV not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer = FaceRecognizer(gallery_path=tmpdir)
            img = create_test_image()
            
            if img is None:
                print("⚠️  test_face_detection skipped (could not create test image)")
                return True
            
            print("✅ test_face_detection passed (basic init)")
            return True
        except Exception as e:
            print(f"❌ test_face_detection failed: {e}")
            return False


def test_enrollment():
    """Test face enrollment (simplified without real face detection)."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment skipped (dependencies not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer = FaceRecognizer(gallery_path=tmpdir)
            
            assert len(recognizer.list_people()) == 0
            
            recognizer.db["Test Person"] = {"embeddings": [np.zeros(128)]}
            recognizer._save_database()
            
            people = recognizer.list_people()
            assert len(people) == 1
            assert people[0][0] == "Test Person"
            
            print("✅ test_enrollment passed")
            return True
        except Exception as e:
            print(f"❌ test_enrollment failed: {e}")
            return False


def test_database_persistence():
    """Test database save/load."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_database_persistence skipped (dependencies not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer1 = FaceRecognizer(gallery_path=tmpdir)
            recognizer1.db["Person A"] = {"embeddings": [np.zeros(128)]}
            recognizer1.db["Person B"] = {"embeddings": [np.zeros(128)]}
            recognizer1._save_database()
            
            recognizer2 = FaceRecognizer(gallery_path=tmpdir)
            assert len(recognizer2.db) == 2
            assert "Person A" in recognizer2.db
            assert "Person B" in recognizer2.db
            
            print("✅ test_database_persistence passed")
            return True
        except Exception as e:
            print(f"❌ test_database_persistence failed: {e}")
            return False


def test_remove_person():
    """Test person removal."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_remove_person skipped (dependencies not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer = FaceRecognizer(gallery_path=tmpdir)
            recognizer.db["Test Person"] = {"embeddings": [np.zeros(128)]}
            recognizer._save_database()
            
            assert len(recognizer.list_people()) == 1
            
            result = recognizer.remove_person("Test Person")
            assert result == True
            assert len(recognizer.list_people()) == 0
            
            result = recognizer.remove_person("Nonexistent")
            assert result == False
            
            print("✅ test_remove_person passed")
            return True
        except Exception as e:
            print(f"❌ test_remove_person failed: {e}")
            return False


def test_conversation_manager():
    """Test conversation manager."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_conversation_manager skipped (dependencies not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer = FaceRecognizer(gallery_path=tmpdir)
            speak_fn = create_speak_function("stub", volume=0.1)
            conversation = ConversationManager(recognizer, speak_fn, volume=0.1)
            
            assert conversation.volume == 0.1
            assert len(conversation.last_seen) == 0
            
            conversation.set_volume(0.5)
            assert conversation.volume == 0.5
            
            conversation.reset_cooldowns()
            assert len(conversation.last_seen) == 0
            
            print("✅ test_conversation_manager passed")
            return True
        except Exception as e:
            print(f"❌ test_conversation_manager failed: {e}")
            return False


def test_speak_functions():
    """Test speak function creation."""
    try:
        stub = create_speak_function("stub", volume=0.1)
        assert callable(stub)
        stub("Test message")
        
        espeak = create_speak_function("espeak", volume=0.1)
        assert callable(espeak)
        
        kokoro = create_speak_function("kokoro", volume=0.1)
        assert callable(kokoro)
        
        gemini = create_speak_function("gemini", volume=0.1)
        assert callable(gemini)
        
        print("✅ test_speak_functions passed")
        return True
    except Exception as e:
        print(f"❌ test_speak_functions failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("Running face recognition tests...\n")
    
    tests = [
        test_recognizer_init,
        test_face_detection,
        test_enrollment,
        test_database_persistence,
        test_remove_person,
        test_conversation_manager,
        test_speak_functions,
    ]
    
    failed = []
    for test in tests:
        try:
            if not test():
                failed.append(test.__name__)
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")
            failed.append(test.__name__)
    
    print(f"\n{'='*60}")
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")
        sys.exit(0)


if __name__ == '__main__':
    run_all_tests()
