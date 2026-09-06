"""CLI for face enrollment and recognition practice.

Usage:
    python -m faces.cli enroll "John Doe" photo.jpg
    python -m faces.cli list
    python -m faces.cli recognize photo.jpg
    python -m faces.cli webcam
    python -m faces.cli remove "John Doe"
"""

import sys
import argparse
from pathlib import Path

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

try:
    from PIL import Image
    import numpy as np
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from faces.recognizer import FaceRecognizer


def load_image(path: str):
    """Load image from path."""
    if not _HAS_OPENCV and not _HAS_PIL:
        print("Error: Neither OpenCV nor PIL available. Install opencv-python or pillow")
        sys.exit(1)
    
    if _HAS_OPENCV:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image: {path}")
            sys.exit(1)
        return img
    elif _HAS_PIL:
        pil_img = Image.open(path)
        return np.array(pil_img)[:, :, ::-1]


def cmd_enroll(args):
    """Enroll a person from image(s)."""
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    
    total = 0
    for img_path in args.images:
        print(f"\nProcessing: {img_path}")
        img = load_image(img_path)
        count = recognizer.enroll(args.name, img)
        total += count
    
    if total > 0:
        print(f"\n✅ Successfully enrolled {total} face(s) for {args.name}")
    else:
        print(f"\n❌ No faces found in images")
        sys.exit(1)


def cmd_list(args):
    """List all enrolled people."""
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    people = recognizer.list_people()
    
    if not people:
        print("No people enrolled yet")
        return
    
    print(f"\n{'Name':<20} {'Faces':<10}")
    print("-" * 30)
    for name, count in sorted(people):
        print(f"{name:<20} {count:<10}")
    print(f"\nTotal: {len(people)} people")


def cmd_recognize(args):
    """Recognize faces in image."""
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    
    img = load_image(args.image)
    results = recognizer.recognize(img, threshold=args.threshold)
    
    if not results:
        print("No faces detected")
        return
    
    print(f"\nDetected {len(results)} face(s):")
    print(f"\n{'Name':<20} {'Confidence':<12} {'Box (x,y,w,h)':<20}")
    print("-" * 52)
    for name, confidence, box in results:
        box_str = f"({box[0]},{box[1]},{box[2]},{box[3]})"
        status = "✅" if name != "unknown" else "❓"
        print(f"{status} {name:<18} {confidence:>5.1%}       {box_str}")
    
    if args.show and _HAS_OPENCV:
        img_copy = img.copy()
        for name, confidence, (x, y, w, h) in results:
            color = (0, 255, 0) if name != "unknown" else (0, 165, 255)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({confidence:.1%})"
            cv2.putText(img_copy, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Recognition Results", img_copy)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_webcam(args):
    """Recognize faces from webcam."""
    if not _HAS_OPENCV:
        print("Error: OpenCV required for webcam. Install opencv-python")
        sys.exit(1)
    
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)
    
    print("Webcam active. Press 'q' to quit, 'c' to capture and enroll")
    enroll_mode = False
    enroll_name = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = recognizer.recognize(frame, threshold=args.threshold)
        
        for name, confidence, (x, y, w, h) in results:
            color = (0, 255, 0) if name != "unknown" else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({confidence:.1%})"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if enroll_mode:
            cv2.putText(frame, f"Enrolling: {enroll_name} (press 'y' to confirm, 'n' to cancel)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Face Recognition (q=quit, c=capture)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not enroll_mode:
            name = input("\nEnter name to enroll: ").strip()
            if name:
                enroll_mode = True
                enroll_name = name
        elif key == ord('y') and enroll_mode:
            count = recognizer.enroll(enroll_name, frame)
            if count > 0:
                print(f"✅ Enrolled {enroll_name}")
            enroll_mode = False
            enroll_name = None
        elif key == ord('n') and enroll_mode:
            print("Enrollment canceled")
            enroll_mode = False
            enroll_name = None
    
    cap.release()
    cv2.destroyAllWindows()


def cmd_remove(args):
    """Remove a person from database."""
    recognizer = FaceRecognizer(gallery_path=args.gallery)
    
    if recognizer.remove_person(args.name):
        print(f"✅ Removed {args.name}")
    else:
        print(f"❌ {args.name} not found in database")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Face recognition CLI")
    parser.add_argument("--gallery", default=None,
                       help="Gallery path (default: ~/.kevin/faces)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a person")
    enroll_parser.add_argument("name", help="Person's name")
    enroll_parser.add_argument("images", nargs="+", help="Image path(s)")
    
    list_parser = subparsers.add_parser("list", help="List enrolled people")
    
    recognize_parser = subparsers.add_parser("recognize", help="Recognize faces in image")
    recognize_parser.add_argument("image", help="Image path")
    recognize_parser.add_argument("--threshold", type=float, default=0.6,
                                 help="Recognition threshold (default: 0.6)")
    recognize_parser.add_argument("--show", action="store_true",
                                 help="Show annotated image")
    
    webcam_parser = subparsers.add_parser("webcam", help="Recognize from webcam")
    webcam_parser.add_argument("--camera", type=int, default=0,
                              help="Camera index (default: 0)")
    webcam_parser.add_argument("--threshold", type=float, default=0.6,
                              help="Recognition threshold (default: 0.6)")
    
    remove_parser = subparsers.add_parser("remove", help="Remove a person")
    remove_parser.add_argument("name", help="Person's name")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "recognize":
        cmd_recognize(args)
    elif args.command == "webcam":
        cmd_webcam(args)
    elif args.command == "remove":
        cmd_remove(args)


if __name__ == "__main__":
    main()
