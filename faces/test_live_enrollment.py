"""Tests for interactive live enrollment system."""

import sys
import time
import tempfile
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


def test_enrollment_unknown_face():
    """Test enrollment prompt for unknown face."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment_unknown_face skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,
                cooldown_seconds=60.0
            )
            
            # Unknown face (confidence = 0.0)
            box = (100, 100, 80, 80)
            action = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                speak=True
            )
            
            # Should prompt for name
            assert action is not None
            assert action.kind == "enrollment_prompt"
            assert len(spoken) == 1
            assert "name" in spoken[0].lower()
            
            print("✅ test_enrollment_unknown_face passed")
            return True
    except Exception as e:
        print(f"❌ test_enrollment_unknown_face failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enrollment_low_confidence_known():
    """Test that low-confidence known faces are treated as unknown."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment_low_confidence_known skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            # Add a known person
            recognizer.db["Alice"] = {"embeddings": [np.random.randn(128).astype(np.float32)]}
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,  # 90% bar
                cooldown_seconds=60.0
            )
            
            # Low confidence match (85% < 90%)
            box = (100, 100, 80, 80)
            action = behavior.on_face_seen(
                name="Alice",
                confidence=0.85,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                speak=True
            )
            
            # Should still prompt (below threshold)
            assert action is not None
            assert action.kind == "enrollment_prompt"
            assert len(spoken) == 1
            
            print("✅ test_enrollment_low_confidence_known passed")
            return True
    except Exception as e:
        print(f"❌ test_enrollment_low_confidence_known failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enrollment_name_collection():
    """Test name collection from ASR."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment_name_collection skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,
                cooldown_seconds=60.0
            )
            
            # Start enrollment
            box = (100, 100, 80, 80)
            action1 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                speak=True
            )
            assert action1.kind == "enrollment_prompt"
            
            # Provide name via ASR
            action2 = behavior.on_transcript(
                "My name is Bob",
                now=1001.0,
                speak=True,
                language="en"
            )
            
            # Should acknowledge name
            assert action2 is not None
            assert action2.kind == "enrollment_name_received"
            assert action2.meta["name"] == "Bob"
            assert len(spoken) >= 2
            assert "bob" in spoken[-1].lower()
            
            print("✅ test_enrollment_name_collection passed")
            return True
    except Exception as e:
        print(f"❌ test_enrollment_name_collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enrollment_timeout():
    """Test enrollment timeout when no name provided."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment_timeout skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,
                cooldown_seconds=60.0
            )
            
            # Start enrollment
            box = (100, 100, 80, 80)
            action1 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                speak=True
            )
            assert action1.kind == "enrollment_prompt"
            
            # Wait past timeout (15s default)
            action2 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1016.0,  # 16s later
                speak=True
            )
            
            # Should timeout and politely leave
            assert action2 is not None
            assert action2.kind == "enrollment_timeout"
            assert len(spoken) >= 2
            # Leave message should be polite
            assert any(word in spoken[-1].lower() for word in ["okay", "alright", "worries", "around"])
            
            print("✅ test_enrollment_timeout passed")
            return True
    except Exception as e:
        print(f"❌ test_enrollment_timeout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enrollment_sample_collection():
    """Test face sample collection during enrollment."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment_sample_collection skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,
                cooldown_seconds=60.0
            )
            
            # Start enrollment
            box = (100, 100, 80, 80)
            action1 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                speak=True
            )
            assert action1.kind == "enrollment_prompt"
            
            # Provide name
            action2 = behavior.on_transcript(
                "I'm Charlie",
                now=1001.0,
                speak=True,
                language="en"
            )
            assert action2.kind == "enrollment_name_received"
            
            # Collect samples
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            for i in range(3):
                action = behavior.on_face_seen(
                    name="unknown",
                    confidence=0.0,
                    box=box,
                    image=img,
                    landmarks=None,
                    now=1002.0 + i * 0.6,
                    speak=True
                )
                
                if i < 2:
                    # Still collecting
                    assert action.kind in ["enrollment_collecting", "enrollment_complete"]
                else:
                    # Should complete after 3 samples
                    assert action.kind == "enrollment_complete"
                    assert "Charlie" in action.utterance
            
            # Verify Charlie was enrolled
            assert "Charlie" in recognizer.db
            
            print("✅ test_enrollment_sample_collection passed")
            return True
    except Exception as e:
        print(f"❌ test_enrollment_sample_collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enrollment_cooldown():
    """Test that unknown faces respect cooldown period."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_enrollment_cooldown skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,
                cooldown_seconds=60.0
            )
            
            # First prompt
            box = (100, 100, 80, 80)
            action1 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                speak=True
            )
            assert action1.kind == "enrollment_prompt"
            initial_spoken = len(spoken)
            
            # Cancel session
            if behavior.enrollment_manager:
                behavior.enrollment_manager.cancel_session()
            
            # Try again immediately (should be in cooldown)
            action2 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1001.0,
                speak=True
            )
            
            # Should NOT prompt again (in cooldown)
            assert action2.kind == "face_unknown"
            assert len(spoken) == initial_spoken  # No new speech
            
            # Try after cooldown expires
            action3 = behavior.on_face_seen(
                name="unknown",
                confidence=0.0,
                box=box,
                image=None,
                landmarks=None,
                now=1061.0,  # 61s later
                speak=True
            )
            
            # Should prompt again
            assert action3.kind == "enrollment_prompt"
            assert len(spoken) > initial_spoken
            
            print("✅ test_enrollment_cooldown passed")
            return True
    except Exception as e:
        print(f"❌ test_enrollment_cooldown failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_high_confidence_skips_enrollment():
    """Test that high-confidence matches skip enrollment and greet normally."""
    if not _HAS_NUMPY or not _HAS_OPENCV:
        print("⚠️  test_high_confidence_skips_enrollment skipped (dependencies not available)")
        return True
    
    try:
        from faces.recognizer import FaceRecognizer
        from faces.people_behavior import create_people_behavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(gallery_path=tmpdir, backend="opencv-dnn")
            
            # Add a known person
            recognizer.db["Alice"] = {"embeddings": [np.random.randn(128).astype(np.float32)]}
            
            spoken = []
            def mock_speak(text):
                spoken.append(text)
            
            behavior = create_people_behavior(
                speak_fn=mock_speak,
                recognizer=recognizer,
                enable_live_enrollment=True,
                enrollment_confidence_threshold=0.90,
                cooldown_seconds=60.0
            )
            
            # High confidence match (95% >= 90%)
            box = (100, 100, 80, 80)
            action = behavior.on_face_seen(
                name="Alice",
                confidence=0.95,
                box=box,
                image=None,
                landmarks=None,
                now=1000.0,
                hour=14,  # 2 PM
                speak=True
            )
            
            # Should greet normally, NOT enrollment
            assert action is not None
            assert action.kind == "greet"
            assert "Alice" in action.utterance
            assert len(spoken) == 1
            assert "alice" in spoken[0].lower()
            
            print("✅ test_high_confidence_skips_enrollment passed")
            return True
    except Exception as e:
        print(f"❌ test_high_confidence_skips_enrollment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all live enrollment tests."""
    print("Running live enrollment tests...\n")
    
    tests = [
        test_enrollment_unknown_face,
        test_enrollment_low_confidence_known,
        test_enrollment_name_collection,
        test_enrollment_timeout,
        test_enrollment_sample_collection,
        test_enrollment_cooldown,
        test_high_confidence_skips_enrollment,
    ]
    
    failed = []
    for test in tests:
        try:
            if not test():
                failed.append(test.__name__)
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
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
