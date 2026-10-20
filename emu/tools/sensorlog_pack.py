"""Pack the raw depth of a sensorlog v4 recording (SENSORLOG_ENC=gpu writes <stamp>.npy, uint16 mm) into the usual
lossless PNG16 (SUB filter, zlib 1): python emu/tools/sensorlog_pack.py <logdir> [threads]
Heavy CPU (~16 ms/frame): run it through emu/tools/offline_gate.sh so it only runs while the sim is stopped.  Each .npy is
replaced only after its PNG is written and read back identical."""
import glob, os, sys
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_FILTER, cv2.IMWRITE_PNG_FILTER_SUB]

def pack(f):
    a = np.load(f); out = f[:-4] + ".png"; tmp = out + ".tmp.png"; cv2.imwrite(tmp, a, PNG)
    if not np.array_equal(cv2.imread(tmp, cv2.IMREAD_UNCHANGED), a): os.unlink(tmp); return f
    os.replace(tmp, out); os.unlink(f); return None

if __name__ == "__main__":
    d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4; cv2.setNumThreads(1)
    fs = sorted(glob.glob(os.path.join(d, "rs*", "*.npy")))
    with ThreadPoolExecutor(n) as ex: bad = [x for x in ex.map(pack, fs) if x]
    print("sensorlog_pack %s: %d raw depth frames -> png, %d mismatches kept as .npy" % (d, len(fs) - len(bad), len(bad)))
