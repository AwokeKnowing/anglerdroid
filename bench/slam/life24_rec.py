"""Record the exact slam2 job stream of a life24 sim wander (for offline replay on other hardware, e.g. the Orin).

python bench/slam/life24_rec.py --rec OUT/jobs.pkl.gz [life24 args...]   (e.g. --seed 5 --hours 1 --rtab 0 --out OUT)
Writes gzip'd pickle records in the KEVIN_SLAM_REC format (t, odom_pose, obs, free, obs_h, rays, T_before), one per
Slam2._step call, i.e. exactly what life24 feeds slam2 (rs2_ir1 depth @5 Hz; no rays/obs_h, like life24).
Replay: bench/slam/orin_bench.py."""
import gzip, os, pickle, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
i = sys.argv.index("--rec"); rec = os.path.expanduser(sys.argv[i + 1]); del sys.argv[i:i + 2]
os.environ.setdefault("KEVIN_SLAM2_LIFE", "1")
if os.environ.get("KEVIN_TORCH_SITE"): sys.path.append(os.environ["KEVIN_TORCH_SITE"])
sys.path[:0] = [REPO, HERE]
import nav.slam2 as S2
f = gzip.open(rec, "wb", compresslevel=3); _step = S2.Slam2._step
def step(self, t, podom, obs, free, obs_h=None, rays=None):
    pickle.dump((t, podom, obs, free, obs_h, rays, tuple(self.T)), f, protocol=4)
    return _step(self, t, podom, obs, free, obs_h, rays)
S2.Slam2._step = step
try:
    sys.argv[0] = os.path.join(HERE, "life24.py"); runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    f.close()
