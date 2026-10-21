"""Disk guard for our generated emulator data (i777).
ok = diskguard.check()  -> True if writing may proceed.
 free >= ROTATE_GB (100): ok.
 below: delete OUR oldest regenerable artifacts (ROTATABLE globs only, never user data), oldest first,
   skipping anything modified in the last 30 min (may be in use), until free >= ROTATE_GB.
 still below HARD_GB (30): refuse (writers stop / skip).
CLI: python diskguard.py [--dry]   (prints free GB and what it would rotate)"""
import os, sys, glob, time, shutil
H = os.path.expanduser("~/.kevin/emu")
ROTATE_GB = float(os.environ.get("KEVIN_DISK_ROTATE_GB", 100)); HARD_GB = float(os.environ.get("KEVIN_DISK_HARD_GB", 30))
ROTATABLE = [H + "/stereo2/*.npz", H + "/stereo/*.npz", H + "/rgbd/run*", os.path.expanduser("~/kevin_clips/*.tar"), "/tmp/orb_bench*.yaml"]
PACKED_FRAMES = H + "/clips/*"          # raw clip frames: rotatable only once a ~/kevin_clips/*_<name>.tar exists
def free_gb(p="/"): return shutil.disk_usage(p).free / 2**30
def _size(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)
def candidates():
    c = [p for g in ROTATABLE for p in glob.glob(g)]
    tars = {os.path.basename(t).split("_", 1)[-1][:-4] for t in glob.glob(os.path.expanduser("~/kevin_clips/*_*.tar"))}
    c += [p for p in glob.glob(PACKED_FRAMES) if os.path.isdir(p) and os.path.basename(p) in tars]
    now = time.time(); return sorted((p for p in c if now - os.path.getmtime(p) > 1800), key=os.path.getmtime)
def check(dry=False, log=print):
    for p in candidates():
        if free_gb() >= ROTATE_GB: break
        log("diskguard: free %.0f GB < %.0f, rotating %s (%.2f GB)" % (free_gb(), ROTATE_GB, p, _size(p) / 2**30))
        if not dry: shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    ok = free_gb() >= HARD_GB
    if not ok: log("diskguard: REFUSE write, free %.0f GB < hard floor %.0f GB" % (free_gb(), HARD_GB))
    return ok
if __name__ == "__main__":
    print("free %.1f GB" % free_gb()); print("ok", check(dry="--dry" in sys.argv))
