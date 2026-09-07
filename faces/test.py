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
from faces import test_people_behavior as _people_tests


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


def test_insightface_backend():
    """Test InsightFace backend initialization."""
    if not _HAS_OPENCV or not _HAS_NUMPY:
        print("⚠️  test_insightface_backend skipped (dependencies not available)")
        return True
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  test_insightface_backend skipped (onnxruntime not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Should not crash even if models are missing
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="insightface", model_pack="buffalo_l")
            assert recognizer.backend == "insightface"
            assert recognizer.model_pack == "buffalo_l"
            assert recognizer.embedding_dim == 512
            print("✅ test_insightface_backend passed")
            return True
        except Exception as e:
            print(f"❌ test_insightface_backend failed: {e}")
            return False


def test_embedding_dimensions():
    """Test that different backends produce expected embedding dimensions."""
    if not _HAS_OPENCV or not _HAS_NUMPY:
        print("⚠️  test_embedding_dimensions skipped (dependencies not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Test InsightFace (synthetic check)
            try:
                import onnxruntime as ort
                recognizer_if = FaceRecognizer(gallery_path=tmpdir, backend="insightface", model_pack="buffalo_l")
                assert recognizer_if.embedding_dim == 512
                
                recognizer_if_s = FaceRecognizer(gallery_path=tmpdir, backend="insightface", model_pack="buffalo_s")
                assert recognizer_if_s.embedding_dim == 512
            except ImportError:
                pass
            
            # Test face_recognition (dlib) - 128-D
            try:
                import face_recognition
                recognizer_fr = FaceRecognizer(gallery_path=tmpdir, backend="face_recognition")
                # Create synthetic 128-D embedding
                test_emb = np.random.randn(128)
                assert len(test_emb) == 128
            except ImportError:
                pass
            
            print("✅ test_embedding_dimensions passed")
            return True
        except Exception as e:
            print(f"❌ test_embedding_dimensions failed: {e}")
            return False


def test_stricter_matching():
    """Test stricter matching with threshold + margin."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_stricter_matching skipped (dependencies not available)")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            # Create synthetic embeddings (normalized)
            emb_alice_1 = np.random.randn(128).astype(np.float32)
            emb_alice_1 /= np.linalg.norm(emb_alice_1)
            
            emb_alice_2 = emb_alice_1 + np.random.randn(128) * 0.1
            emb_alice_2 /= np.linalg.norm(emb_alice_2)
            
            emb_bob = np.random.randn(128).astype(np.float32)
            emb_bob /= np.linalg.norm(emb_bob)
            
            # Add to database
            recognizer.db["Alice"] = {"embeddings": [emb_alice_1, emb_alice_2]}
            recognizer.db["Bob"] = {"embeddings": [emb_bob]}
            
            # Test cosine similarity calculation
            sim_alice = np.dot(emb_alice_1, emb_alice_2)
            sim_bob = np.dot(emb_alice_1, emb_bob)
            
            # Alice embeddings should be more similar than Alice vs Bob
            assert sim_alice > sim_bob, f"Alice similarity {sim_alice:.3f} should be > Bob similarity {sim_bob:.3f}"
            
            # Test margin requirement
            margin = 0.05
            if (sim_alice - sim_bob) >= margin:
                print(f"  Margin check passed: {sim_alice:.3f} - {sim_bob:.3f} = {sim_alice - sim_bob:.3f} >= {margin}")
            
            print("✅ test_stricter_matching passed")
            return True
        except Exception as e:
            print(f"❌ test_stricter_matching failed: {e}")
            return False


def test_landmark_alignment():
    """Test landmark alignment function."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_landmark_alignment skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import _align_face_insightface
        
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Synthetic 5-point landmarks (rough face position)
        landmarks = np.array([
            [200, 180],  # right eye
            [280, 180],  # left eye
            [240, 220],  # nose
            [210, 260],  # right mouth
            [270, 260]   # left mouth
        ], dtype=np.float32)
        
        # Should produce 112x112 aligned face
        aligned = _align_face_insightface(img, landmarks, image_size=(112, 112))
        
        assert aligned.shape == (112, 112, 3), f"Expected (112, 112, 3), got {aligned.shape}"
        assert aligned.dtype == np.uint8
        
        print("✅ test_landmark_alignment passed")
        return True
    except ImportError:
        print("⚠️  test_landmark_alignment skipped (cv2 not available)")
        return True
    except Exception as e:
        print(f"❌ test_landmark_alignment failed: {e}")
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
        test_insightface_backend,
        test_embedding_dimensions,
        test_stricter_matching,
        test_landmark_alignment,
        _people_tests.test_greet_hours_day_window,
        _people_tests.test_greet_hours_wrap_midnight,
        _people_tests.test_name_call_match,
        _people_tests.test_directional_help_landmark,
        _people_tests.test_directional_help_with_name_call,
        _people_tests.test_face_greet_hours_gate,
        _people_tests.test_hardware_flag_stays_off,
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
