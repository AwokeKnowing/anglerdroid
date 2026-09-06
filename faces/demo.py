"""Demo script for face recognition with conversation.

Demonstrates:
- Face detection and recognition
- Greeting known people
- Asking names of unknown people
- Enrollment workflow

Usage:
    python -m faces.demo --webcam       # Live webcam demo
    python -m faces.demo --images dir/  # Process directory of images
"""

import sys
import argparse
import time
from pathlib import Path

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from faces.recognizer import FaceRecognizer
from faces.conversation import ConversationManager, create_speak_function


def demo_webcam(args):
    """Run live webcam demo with conversation."""
    if not _HAS_OPENCV:
        print("Error: OpenCV required for webcam. Install opencv-python")
        sys.exit(1)
    
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    speak_fn = create_speak_function(args.tts, volume=args.volume)
    conversation = ConversationManager(recognizer, speak_fn, volume=args.volume)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("FACE RECOGNITION DEMO - Webcam Mode")
    print("=" * 60)
    print("Controls:")
    print("  q - Quit")
    print("  e - Enroll unknown face (will prompt for name)")
    print("  r - Reset cooldowns (allow immediate re-greeting)")
    print("  v - Toggle volume")
    print("=" * 60 + "\n")
    
    enroll_mode = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0 or enroll_mode:
            result = conversation.process_frame(frame)
            
            display_frame = frame.copy()
            for name, confidence, (x, y, w, h) in result["faces"]:
                color = (0, 255, 0) if name != "unknown" else (0, 165, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name} ({confidence:.1%})"
                cv2.putText(display_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if enroll_mode:
                cv2.putText(display_frame, "ENROLL MODE - Enter name in terminal",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"Vol: {args.volume:.0%}", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Face Recognition Demo (q=quit)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            enroll_mode = True
            name = input("\nEnter name to enroll: ").strip()
            if name:
                if conversation.enroll_from_input(frame, name):
                    print(f"✅ Enrolled {name}")
                else:
                    print(f"❌ No face detected to enroll")
            enroll_mode = False
        elif key == ord('r'):
            conversation.reset_cooldowns()
            print("✅ Cooldowns reset - greetings will trigger again")
        elif key == ord('v'):
            args.volume = 0.5 if args.volume < 0.5 else 0.1
            conversation.set_volume(args.volume)
    
    cap.release()
    cv2.destroyAllWindows()


def demo_images(args):
    """Process directory of images."""
    if not _HAS_OPENCV:
        print("Error: OpenCV required. Install opencv-python")
        sys.exit(1)
    
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    speak_fn = create_speak_function(args.tts, volume=args.volume)
    conversation = ConversationManager(recognizer, speak_fn, volume=args.volume)
    
    image_dir = Path(args.images)
    if not image_dir.is_dir():
        print(f"Error: Not a directory: {args.images}")
        sys.exit(1)
    
    image_files = list(image_dir.glob("*.jpg")) + \
                  list(image_dir.glob("*.png")) + \
                  list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {args.images}")
        sys.exit(1)
    
    print(f"\nProcessing {len(image_files)} images from {args.images}\n")
    
    for img_path in sorted(image_files):
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print('='*60)
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load image")
            continue
        
        result = conversation.process_frame(img)
        
        if result["faces"]:
            print(f"\nDetected {len(result['faces'])} face(s):")
            for name, confidence, _ in result["faces"]:
                status = "✅" if name != "unknown" else "❓"
                print(f"  {status} {name} ({confidence:.1%})")
        else:
            print("No faces detected")
        
        if result["greetings"]:
            print(f"\nGreetings sent: {len(result['greetings'])}")
        
        if result["unknowns"]:
            print(f"\nUnknown faces: {len(result['unknowns'])}")
        
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Face recognition demo")
    parser.add_argument("--gallery", default=None,
                       help="Gallery path (default: ~/.kevin/faces)")
    parser.add_argument("--tts", default="stub",
                       choices=["stub", "espeak", "kokoro", "gemini"],
                       help="TTS backend (default: stub)")
    parser.add_argument("--volume", type=float, default=0.1,
                       help="TTS volume 0.0-1.0 (default: 0.1 = 10%%)")
    
    subparsers = parser.add_subparsers(dest="mode", help="Demo mode")
    
    webcam_parser = subparsers.add_parser("webcam", help="Live webcam demo")
    webcam_parser.add_argument("--camera", type=int, default=0,
                              help="Camera index (default: 0)")
    
    images_parser = subparsers.add_parser("images", help="Process image directory")
    images_parser.add_argument("dir", metavar="images",
                              help="Directory containing images")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    if args.mode == "webcam":
        demo_webcam(args)
    elif args.mode == "images":
        demo_images(args)


if __name__ == "__main__":
    main()
