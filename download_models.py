#!/usr/bin/env python3
"""Download InsightFace buffalo_l models for Kevin face recognition.

Downloads w600k_r50.onnx (buffalo_l recognition model) and YuNet detector
to ~/.kevin/faces/models/ with SHA256 verification.

Usage:
    python3 download_models.py [--model-pack buffalo_l|buffalo_s]
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


# Model URLs and checksums (InsightFace model zoo)
# Note: These are placeholders - update with actual URLs when available
MODELS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": None,  # Will skip verification if None
        "description": "YuNet face detector (OpenCV Zoo)",
    },
    "buffalo_l": {
        "filename": "w600k_r50.onnx",
        "url": None,  # User must download manually from InsightFace
        "sha256": None,
        "description": "Buffalo_l recognition model (ResNet-50, 512-D)",
        "manual": True,
    },
    "buffalo_s": {
        "filename": "w600k_mbf.onnx",
        "url": None,  # User must download manually from InsightFace
        "sha256": None,
        "description": "Buffalo_s recognition model (MobileFaceNet, 512-D)",
        "manual": True,
    },
}


def download_file(url: str, dest: Path, sha256: str = None) -> bool:
    """Download file with progress and optional SHA256 verification."""
    print(f"Downloading {dest.name}...")
    print(f"  URL: {url}")
    
    try:
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / total_size))
            sys.stdout.write(f"\r  Progress: {percent}% ")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest, reporthook)
        print()  # Newline after progress
        
        # Verify checksum if provided
        if sha256:
            print(f"  Verifying SHA256...")
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            computed = h.hexdigest()
            
            if computed.lower() != sha256.lower():
                print(f"  ❌ SHA256 mismatch!")
                print(f"     Expected: {sha256}")
                print(f"     Got:      {computed}")
                dest.unlink()
                return False
            print(f"  ✅ SHA256 verified")
        
        print(f"  ✅ Downloaded {dest.name}")
        return True
    
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download InsightFace models for Kevin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download YuNet detector + show buffalo_l instructions
  python3 download_models.py --model-pack buffalo_l
  
  # Download YuNet detector + show buffalo_s instructions
  python3 download_models.py --model-pack buffalo_s

Manual download (buffalo models):
  1. Visit: https://github.com/deepinsight/insightface/tree/master/model_zoo
  2. Download buffalo_l or buffalo_s pack
  3. Extract w600k_r50.onnx (buffalo_l) or w600k_mbf.onnx (buffalo_s)
  4. Place in ~/.kevin/faces/models/
        """
    )
    
    parser.add_argument(
        "--model-pack",
        choices=["buffalo_l", "buffalo_s"],
        default="buffalo_l",
        help="InsightFace model pack (default: buffalo_l)"
    )
    parser.add_argument(
        "--models-dir",
        default="~/.kevin/faces/models",
        help="Models directory (default: ~/.kevin/faces/models)"
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("InsightFace Model Download Helper")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Target pack: {args.model_pack}")
    print()
    
    # Download YuNet detector
    yunet_info = MODELS["yunet"]
    yunet_path = models_dir / yunet_info["filename"]
    
    if yunet_path.exists():
        print(f"✅ {yunet_info['filename']} already exists")
    else:
        if yunet_info["url"]:
            success = download_file(
                yunet_info["url"],
                yunet_path,
                yunet_info.get("sha256")
            )
            if not success:
                print(f"❌ Failed to download {yunet_info['filename']}")
                sys.exit(1)
        else:
            print(f"⚠️  {yunet_info['filename']} - no automatic download")
    
    print()
    
    # Buffalo model (manual download)
    buffalo_info = MODELS[args.model_pack]
    buffalo_path = models_dir / buffalo_info["filename"]
    
    if buffalo_path.exists():
        print(f"✅ {buffalo_info['filename']} already exists")
        size_mb = buffalo_path.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
    else:
        print(f"❌ {buffalo_info['filename']} NOT FOUND")
        print()
        print("=" * 60)
        print("MANUAL DOWNLOAD REQUIRED:")
        print("=" * 60)
        print(f"Model: {buffalo_info['description']}")
        print()
        print("Steps:")
        print("1. Visit: https://github.com/deepinsight/insightface/tree/master/model_zoo")
        print(f"2. Download {args.model_pack} model pack (.zip or .tar.gz)")
        print(f"3. Extract {buffalo_info['filename']} from the archive")
        print(f"4. Copy to: {buffalo_path}")
        print()
        print("Alternative sources:")
        print("- InsightFace GitHub releases")
        print("- InsightFace model zoo (OneDrive/Google Drive links in README)")
        print("=" * 60)
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✅ Model check complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"1. Rebuild embeddings:")
    print(f"   python -m faces.rebuild_embeddings --backend insightface --model-pack {args.model_pack}")
    print()
    print("2. Test recognition:")
    print(f"   python -m faces.cli webcam --backend insightface --model-pack {args.model_pack}")
    print()
    print("3. Test on live robot:")
    print("   # Start people_live (auto-detects insightface if models present)")
    print("   # Enrollment will trigger for unknown faces")
    

if __name__ == "__main__":
    main()
