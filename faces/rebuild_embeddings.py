"""Rebuild face embeddings database from existing gallery crops.

This script re-processes all existing face crops in the gallery
using a new/different backend (e.g., upgrading from SFace to InsightFace).

Usage:
    python -m faces.rebuild_embeddings --backend insightface --model-pack buffalo_l
    python -m faces.rebuild_embeddings --backend insightface --model-pack buffalo_s
    python -m faces.rebuild_embeddings --gallery ~/.kevin/faces --backup
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False
    print("Error: opencv-python required. Install with: pip install opencv-python")
    sys.exit(1)

from faces.recognizer import FaceRecognizer


def backup_database(gallery_path: Path) -> None:
    """Backup existing database.pkl before rebuilding."""
    db_path = gallery_path / "database.pkl"
    if db_path.exists():
        backup_path = gallery_path / "database.pkl.backup"
        counter = 1
        while backup_path.exists():
            backup_path = gallery_path / f"database.pkl.backup.{counter}"
            counter += 1
        shutil.copy(db_path, backup_path)
        print(f"✅ Backed up database to {backup_path}")


def rebuild_embeddings(gallery_path: str, backend: str, model_pack: str, backup: bool = True) -> None:
    """Rebuild all embeddings from existing gallery crops.
    
    Args:
        gallery_path: Path to gallery (e.g., ~/.kevin/faces)
        backend: Recognition backend (insightface, face_recognition, opencv-dnn)
        model_pack: For insightface: buffalo_l or buffalo_s
        backup: Whether to backup existing database.pkl
    """
    gallery = Path(gallery_path).expanduser()
    
    if not gallery.exists():
        print(f"Error: Gallery path does not exist: {gallery}")
        sys.exit(1)
    
    # Check existing database for dimension mismatch
    db_path = gallery / "database.pkl"
    if db_path.exists():
        try:
            import pickle
            with open(db_path, "rb") as f:
                old_db = pickle.load(f)
            
            # Check embedding dimensions
            if old_db:
                for name, data in old_db.items():
                    if "embeddings" in data and data["embeddings"]:
                        old_dim = len(data["embeddings"][0])
                        
                        # InsightFace uses 512-D, face_recognition uses 128-D, opencv varies
                        expected_dim = None
                        if backend == "insightface":
                            expected_dim = 512
                        elif backend == "face_recognition":
                            expected_dim = 128
                        
                        if expected_dim and old_dim != expected_dim:
                            print()
                            print("⚠️  WARNING: Embedding dimension mismatch detected!")
                            print(f"   Existing database: {old_dim}-D embeddings")
                            print(f"   New backend ({backend}): {expected_dim}-D embeddings")
                            print()
                            print("The old database MUST be rebuilt with the new backend.")
                            print("Mixing different embedding dimensions will cause recognition failures.")
                            print()
                            if not backup:
                                print("❌ Cannot proceed without backup enabled.")
                                print("   Run with --backup (default) to backup old database first.")
                                sys.exit(1)
                        break
        except Exception as e:
            print(f"Warning: Could not check existing database: {e}")
    
    # Backup existing database if requested
    if backup:
        backup_database(gallery)
    
    # Create new recognizer with specified backend
    print(f"\n🔧 Initializing {backend} backend (model_pack={model_pack})...")
    recognizer = FaceRecognizer(gallery_path=str(gallery), backend=backend, model_pack=model_pack)
    
    # Clear existing database (start fresh with new dimensions)
    recognizer.db = {}
    
    # Find all person directories
    person_dirs = [d for d in gallery.iterdir() if d.is_dir() and d.name != "models"]
    
    if not person_dirs:
        print(f"⚠️  No person directories found in {gallery}")
        print("Gallery structure should be: ~/.kevin/faces/<person_name>/*.jpg")
        return
    
    print(f"\n📂 Found {len(person_dirs)} person directories")
    print("-" * 60)
    
    total_images = 0
    total_embeddings = 0
    
    for person_dir in sorted(person_dirs):
        # Convert directory name back to display name
        person_name = person_dir.name.replace("_", " ").title()
        
        # Find all images
        image_paths = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.jpeg")) + sorted(person_dir.glob("*.png"))
        
        if not image_paths:
            print(f"⚠️  {person_name}: No images found, skipping")
            continue
        
        print(f"\n👤 Processing {person_name} ({len(image_paths)} images)...")
        
        person_embeddings = 0
        for img_path in image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ⚠️  Could not load {img_path.name}, skipping")
                continue
            
            # Detect and embed (should find exactly 1 face per crop)
            count = recognizer.enroll(person_name, img)
            person_embeddings += count
            total_images += 1
            
            if count == 0:
                print(f"  ⚠️  {img_path.name}: No face detected")
            elif count > 1:
                print(f"  ℹ️  {img_path.name}: Multiple faces detected ({count})")
        
        total_embeddings += person_embeddings
        print(f"  ✅ {person_name}: {person_embeddings} embeddings extracted")
    
    # Report embedding dimension
    if recognizer.db:
        first_person = list(recognizer.db.keys())[0]
        if recognizer.db[first_person]["embeddings"]:
            emb_dim = len(recognizer.db[first_person]["embeddings"][0])
            print(f"\n📏 Embedding dimension: {emb_dim}-D")
    
    print("\n" + "=" * 60)
    print(f"✅ Rebuild complete!")
    print(f"   Processed: {total_images} images")
    print(f"   Extracted: {total_embeddings} embeddings")
    print(f"   People: {len(recognizer.db)}")
    print(f"   Backend: {backend}")
    if backend == "insightface":
        print(f"   Model pack: {model_pack}")
        if hasattr(recognizer, 'embedding_dim'):
            print(f"   Embedding dim: {recognizer.embedding_dim}-D")
    print(f"   Database: {gallery / 'database.pkl'}")
    print("=" * 60)
    
    # Show summary
    print("\n📊 Summary:")
    for name, data in sorted(recognizer.db.items()):
        print(f"  {name}: {len(data['embeddings'])} embeddings")
    
    print("\n⚠️  IMPORTANT: Old SFace/128-D embeddings are incompatible with new 512-D embeddings.")
    print("If you see recognition failures, ensure all embeddings were rebuilt with the same backend.")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild face embeddings database from existing gallery crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rebuild with InsightFace buffalo_l (512-D, best quality)
  python -m faces.rebuild_embeddings --backend insightface --model-pack buffalo_l

  # Rebuild with InsightFace buffalo_s (mobile)
  python -m faces.rebuild_embeddings --backend insightface --model-pack buffalo_s

  # Rebuild with face_recognition (dlib)
  python -m faces.rebuild_embeddings --backend face_recognition

  # Custom gallery path
  python -m faces.rebuild_embeddings --gallery /path/to/gallery --backend insightface
        """
    )
    
    parser.add_argument("--gallery", default="~/.kevin/faces",
                       help="Gallery path (default: ~/.kevin/faces)")
    parser.add_argument("--backend", default="insightface",
                       choices=["insightface", "face_recognition", "opencv-dnn"],
                       help="Recognition backend (default: insightface)")
    parser.add_argument("--model-pack", default="buffalo_l",
                       choices=["buffalo_l", "buffalo_s"],
                       help="InsightFace model pack (default: buffalo_l)")
    parser.add_argument("--no-backup", dest="backup", action="store_false",
                       help="Skip backing up existing database.pkl")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 Face Embeddings Rebuild Tool")
    print("=" * 60)
    
    rebuild_embeddings(
        gallery_path=args.gallery,
        backend=args.backend,
        model_pack=args.model_pack,
        backup=args.backup
    )


if __name__ == "__main__":
    main()
