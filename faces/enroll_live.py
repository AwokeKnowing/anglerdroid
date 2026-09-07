"""Live webcam enrollment helper for domain gap mitigation.

This tool helps improve recognition by enrolling face crops captured
directly from the robot's webcam, matching the live domain (lighting, 
resolution, compression) rather than phone/DSLR gallery photos.

Usage:
    python -m faces.enroll_live --name "James" --backend insightface
    python -m faces.enroll_live --name "Erika" --samples 5 --show-scores
"""

import argparse
import sys
import time
from typing import Optional

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False
    print("Error: opencv-python required. Install with: pip install opencv-python")
    sys.exit(1)

from faces.recognizer import FaceRecognizer


def capture_live_samples(
    name: str,
    backend: str = "insightface",
    model_pack: str = "buffalo_l",
    gallery_path: Optional[str] = None,
    camera_index: int = 0,
    num_samples: int = 3,
    min_face_size: int = 80,
    show_scores: bool = False
) -> int:
    """Capture live face samples from webcam and enroll them.
    
    Args:
        name: Person's display name
        backend: Recognition backend
        model_pack: InsightFace model pack
        gallery_path: Gallery path (default: ~/.kevin/faces)
        camera_index: Camera device index
        num_samples: Number of samples to capture
        min_face_size: Minimum face size (pixels) to accept
        show_scores: Show recognition scores before enrollment
    
    Returns:
        Number of samples successfully enrolled
    """
    recognizer = FaceRecognizer(gallery_path=gallery_path, backend=backend, model_pack=model_pack)
    
    # Check if person already exists
    if name in recognizer.db:
        existing_count = len(recognizer.db[name]["embeddings"])
        print(f"ℹ️  {name} already has {existing_count} embeddings in gallery")
        response = input("Add more samples? [y/n]: ").strip().lower()
        if response != 'y':
            print("Enrollment canceled.")
            return 0
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return 0
    
    # Set camera resolution (match live robot setup if known)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 60)
    print("📷 Live Enrollment - Capture Instructions")
    print("=" * 60)
    print(f"Person: {name}")
    print(f"Samples needed: {num_samples}")
    print(f"Backend: {backend} ({model_pack})")
    print()
    print("Tips for best results:")
    print("  • Position face at conversational distance (~1-2 meters)")
    print("  • Ensure good lighting (face clearly visible)")
    print("  • Try different angles: front, slight left, slight right")
    print("  • Natural expression (not forced smile)")
    print()
    print("Controls:")
    print("  SPACE - Capture sample")
    print("  Q - Quit without saving")
    print("  ESC - Cancel")
    print("=" * 60)
    print()
    
    samples_captured = 0
    frame_count = 0
    last_capture_time = 0
    
    while samples_captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break
        
        frame_count += 1
        display = frame.copy()
        
        # Detect faces in preview
        if backend == "insightface":
            detections = recognizer.detect_faces_with_landmarks(frame)
            faces = [(box, lm) for box, lm in detections]
        else:
            boxes = recognizer.detect_faces(frame)
            faces = [(box, None) for box in boxes]
        
        # Draw detections
        for box, landmarks in faces:
            x, y, w, h = box
            
            # Check face size
            face_ok = w >= min_face_size and h >= min_face_size
            color = (0, 255, 0) if face_ok else (0, 165, 255)
            thickness = 2 if face_ok else 1
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            
            # Draw landmarks if available
            if landmarks is not None:
                for lx, ly in landmarks:
                    cv2.circle(display, (int(lx), int(ly)), 2, (255, 0, 0), -1)
            
            # Show face size
            cv2.putText(display, f"{w}x{h}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status overlay
        status_y = 30
        cv2.putText(display, f"Samples: {samples_captured}/{num_samples}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(faces) == 0:
            cv2.putText(display, "No face detected", (10, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(faces) > 1:
            cv2.putText(display, f"Multiple faces ({len(faces)}) - capture largest", (10, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            box, _ = faces[0]
            w, h = box[2], box[3]
            if w < min_face_size or h < min_face_size:
                cv2.putText(display, f"Face too small - move closer", (10, status_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(display, "Ready - press SPACE to capture", (10, status_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Live Enrollment", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            print("\n❌ Enrollment canceled by user")
            cap.release()
            cv2.destroyAllWindows()
            return samples_captured
        
        elif key == ord(' '):  # SPACE
            # Debounce captures
            current_time = time.time()
            if current_time - last_capture_time < 0.5:
                continue
            last_capture_time = current_time
            
            if len(faces) == 0:
                print("⚠️  No face detected - capture skipped")
                continue
            
            # Check face size
            box, landmarks = faces[0] if len(faces) == 1 else max(faces, key=lambda f: f[0][2] * f[0][3])
            w, h = box[2], box[3]
            
            if w < min_face_size or h < min_face_size:
                print(f"⚠️  Face too small ({w}x{h}) - move closer (min: {min_face_size}x{min_face_size})")
                continue
            
            # Show recognition score before enrollment (if person exists)
            if show_scores and name in recognizer.db:
                results = recognizer.recognize(frame, threshold=0.0, log_scores=False)
                if results:
                    recognized_name, score, _ = results[0]
                    print(f"📊 Current recognition: {recognized_name} (score: {score:.3f})")
            
            # Enroll the frame
            count = recognizer.enroll(name, frame)
            
            if count > 0:
                samples_captured += count
                print(f"✅ Sample {samples_captured}/{num_samples} captured!")
                
                # Flash effect
                flash = display * 0.7
                flash = flash.astype('uint8')
                cv2.imshow("Live Enrollment", flash)
                cv2.waitKey(100)
            else:
                print("⚠️  Failed to extract embedding - try again")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"✅ Enrollment complete!")
    print(f"   Captured: {samples_captured} samples")
    print(f"   Person: {name}")
    print(f"   Total embeddings: {len(recognizer.db[name]['embeddings'])}")
    print("=" * 60)
    
    return samples_captured


def main():
    parser = argparse.ArgumentParser(
        description="Live webcam enrollment for domain gap mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll new person with 3 samples
  python -m faces.enroll_live --name "James"

  # Add more samples to existing person
  python -m faces.enroll_live --name "Erika" --samples 5

  # Show recognition scores before enrollment
  python -m faces.enroll_live --name "David" --show-scores
        """
    )
    
    parser.add_argument("--name", required=True,
                       help="Person's display name")
    parser.add_argument("--backend", default="insightface",
                       choices=["insightface", "face_recognition", "opencv-dnn"],
                       help="Recognition backend (default: insightface)")
    parser.add_argument("--model-pack", default="buffalo_l",
                       choices=["buffalo_l", "buffalo_s"],
                       help="InsightFace model pack (default: buffalo_l)")
    parser.add_argument("--gallery", default=None,
                       help="Gallery path (default: ~/.kevin/faces)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index (default: 0)")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples to capture (default: 3)")
    parser.add_argument("--min-face-size", type=int, default=80,
                       help="Minimum face size in pixels (default: 80)")
    parser.add_argument("--show-scores", action="store_true",
                       help="Show recognition scores before enrollment")
    
    args = parser.parse_args()
    
    count = capture_live_samples(
        name=args.name,
        backend=args.backend,
        model_pack=args.model_pack,
        gallery_path=args.gallery,
        camera_index=args.camera,
        num_samples=args.samples,
        min_face_size=args.min_face_size,
        show_scores=args.show_scores
    )
    
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
