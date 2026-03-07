#!/usr/bin/env python3
"""
Debug the forward RGB camera (e.g. /dev/video12). Lists devices, tries open by path and by index, reads frames.
Usage: python3 debug_camera.py [/dev/video12]
"""

import sys
import os
import re
import cv2

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video12"
    print("=== Video devices ===")
    if os.path.isdir("/dev"):
        for name in sorted(os.listdir("/dev")):
            if name.startswith("video"):
                path = os.path.join("/dev", name)
                if os.path.exists(path):
                    print("  ", path)
    print()

    # Try to get numeric index from path (e.g. /dev/video12 -> 12)
    index_from_path = None
    m = re.search(r"video(\d+)$", device)
    if m:
        index_from_path = int(m.group(1))
        print("Index from path:", index_from_path)

    print("\n--- Open by path:", device, "---")
    cap_path = cv2.VideoCapture(device, cv2.CAP_V4L2)
    ok_path = cap_path.isOpened()
    print("  isOpened():", ok_path)
    if ok_path:
        w = int(cap_path.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_path.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("  resolution:", w, "x", h)
        ret, frame = cap_path.read()
        print("  read() success:", ret, "shape:", frame.shape if ret else None)
        if ret:
            print("  frame dtype:", frame.dtype)
        cap_path.release()
    else:
        cap_path.release()

    if index_from_path is not None:
        print("\n--- Open by index:", index_from_path, "---")
        cap_idx = cv2.VideoCapture(index_from_path, cv2.CAP_V4L2)
        ok_idx = cap_idx.isOpened()
        print("  isOpened():", ok_idx)
        if ok_idx:
            w = int(cap_idx.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_idx.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("  resolution:", w, "x", h)
            ret, frame = cap_idx.read()
            print("  read() success:", ret, "shape:", frame.shape if ret else None)
            if ret:
                print("  frame dtype:", frame.dtype)
            cap_idx.release()
        else:
            cap_idx.release()

    print("\n--- Try indices 0..15 ---")
    for i in range(16):
        c = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if c.isOpened():
            r, f = c.read()
            c.release()
            print("  ", i, "open, read:", r, ("shape " + str(f.shape) if r else ""))
        else:
            c.release()

    print("\nDone. If path fails but index works, use --rgb1 12 in main.py (vision will try index as fallback).")

if __name__ == "__main__":
    main()
