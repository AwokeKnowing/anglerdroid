"""Sim depth cameras in their own process, like two hardware devices (python -m emu.depthcam <cfg.json>; started by
emu.world when KEVIN_EMU_DEPTHPROC=1, the default).

Why (2026-10-17 latency trace, emu/tools/latrace.py in the safety-core branch): with RS1+RS2 depth rendered one after the
other in a thread of the world process, a frame reached the safety core 13-45 ms (p50) after capture and up to
124 ms (RS2 under VLN load): the render thread shared the GIL with physics, the colour lane and the viewer, RS2 waited
for RS1 (11-15 ms each), and under GPU/CPU load the two renders (~31 ms) filled the 33 ms period so ticks queued.
A real D435 does none of that: both imagers expose at the same instant, frames arrive at a steady 33.3 ms with a fixed
readout latency (measured on Kevin's Orin, D435i 640x480 USB2: capture -> kernel 31.2 ms p50, 0 drops).

Here: the world's physics loop writes the pose of every camera tick (30 Hz, stamp = physics capture time) to shm
kevin_campose; this process (own GIL, pinned CPUs via KEVIN_EMU_DEPTH_CPUS) renders RS1 and RS2 depth IN PARALLEL in one
thread per camera (own GL context each), applies the D4xx noise model, and is the only writer of kevin_rs1/kevin_rs2.
Colour/IR from the world's slow lane (kevin_rs1_side/kevin_rs2_side) ride along with the next depth frame, as before.
KEVIN_EMU_CAM_LAT_MS (default 0): fixed capture->delivery latency; a frame is published no earlier than stamp + latency
(31 = the measured real readout).  Late frames are never held back further; a tick that is already superseded by a
newer pose when its render starts is dropped (counted in the log), exactly like a hardware frame drop."""
from __future__ import annotations
import json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from emu import shm  # noqa: E402


def depth_model(mj, m, Rd, p, seed):
    """D4xx stereo depth model on a depth render (moved from emu/world.py _render_loop, unchanged): returns depth(dr, cam) -> u16 mm."""
    DH, DW = p["DH"], p["DW"]; rng = np.random.default_rng(seed)
    NB = 8   # pre-generated noise banks (generating 400k normals per frame costs more than the render)
    zn = [rng.standard_normal((DH, DW), dtype=np.float32) for _ in range(NB)]
    drop = [rng.random((DH, DW), dtype=np.float32) < p["holes"] for _ in range(NB)]
    k = [0]
    # D4xx stereo depth model (Intel D400 datasheet 337029; "Tuning depth cameras for best performance" whitepaper):
    #   depth origin = LEFT IR imager; RMS error = Z^2 * subpixel / (f_px * B), subpixel ~0.08, B = 50 mm;
    #   invalid band on the LEFT edge, width = f_px * B / Z px (left imager sees what the right one cannot);
    #   min-Z ~0.175 m @848x480 (max disparity), holes at occlusion edges, random dropouts (dark/shiny).
    fpx, B = p["fpx"], p["baseline"]; sub = p["subpix"]; cols = np.arange(DW, dtype=np.float32)[None, :]
    # Sunlight (KEVIN_EMU_SUN=1, outdoor): the 850 nm projector is washed out -> passive stereo only.  D4xx
    # still ranges on textured ground, but subpixel error roughly doubles and bright low-texture surfaces
    # (sunlit concrete, white paint, sky glare) drop out in blobs; whole frames occasionally lose more.
    SUN = os.environ.get("KEVIN_EMU_SUN", "0") == "1"
    if SUN:
        from emu import envs
        sub = sub * 2.0
        sunm = [envs._smooth(rng, (DH, DW), 40) > rng.uniform(0.25, 0.45) for _ in range(NB)]
    t_render = [0]
    def depth(dr, cam):
        Rd.update_scene(dr, cam); Rd.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0   # depth needs no shading passes
        z = Rd.render(); t_render[0] = time.monotonic_ns(); i = k[0] = (k[0] + 1) % NB
        disp = np.float32(fpx * B) / np.maximum(z, np.float32(1e-3))        # disparity (px), reused below
        zz = z * z; zz *= np.float32(sub / (fpx * B)); zz *= zn[i]; zz += z
        bad = (z < p["min_z"]) | (z > 8.0) | (drop[i] if p["holes"] > 0 else False)
        if SUN: bad |= sunm[i] | (rng.random(z.shape) < 0.03)
        if p["band"]: bad |= cols < disp                                         # left invalid band
        # stereo occlusion shadow (left-imager origin): pixel c is unseen by the right imager if a nearer pixel
        # c+s to its right has disparity >= d(c)+s.  Geometric, one-sided, width f*B*(1/zn-1/zf) px.
        xr = cols - disp                                                           # right-image column
        smin = np.minimum.accumulate(xr[:, ::-1], axis=1)[:, ::-1]                 # min over c' >= c
        bad[:, :-1] |= smin[:, 1:] <= xr[:, :-1] - 0.5                             # something to the right lands left of us
        zz *= 1000.0; zz[bad] = 0
        return zz.astype(np.uint16)
    depth.t_render = t_render
    return depth


def pose_layout(m):
    return [("qpos", (m.nq,), "f8"), ("mpos", (max(m.nmocap, 1), 3), "f8"), ("mquat", (max(m.nmocap, 1), 4), "f8"), ("blind", (2,), "u1")]


def side_layout(p):
    L = [("color", (p["CH"], p["CW"], 3), "u1")]
    if p["ir"]: L += [("ir1", (p["DH"], p["DW"]), "u1"), ("ir2", (p["DH"], p["DW"]), "u1")]
    return L


def main(cfg_path):
    p = json.load(open(cfg_path))
    cpus = os.environ.get("KEVIN_EMU_DEPTH_CPUS")
    if cpus:
        try: os.sched_setaffinity(0, {int(c) for c in cpus.split(",")})
        except Exception as e: print("emu.depthcam: affinity %s failed: %s" % (cpus, e), flush=True)
    import mujoco as mj
    m = mj.MjModel.from_binary_path(p["model"])
    pose = shm.Block("kevin_campose", pose_layout(m))
    lat_ns = int(float(os.environ.get("KEVIN_EMU_CAM_LAT_MS", "0")) * 1e6)
    jit = float(p.get("cam_jit", 0.0)); world_pid = p["world_pid"]
    log = open(os.environ["KEVIN_EMU_CAMLOG"] + ".depthcam", "a", buffering=1 << 16) if os.environ.get("KEVIN_EMU_CAMLOG") else None
    llock = threading.Lock(); ready = threading.Barrier(3)
    print("emu.depthcam: pid %d cpus %s latency %.1f ms, parallel RS1/RS2 depth" % (os.getpid(), sorted(os.sched_getaffinity(0)), lat_ns * 1e-6), flush=True)

    def lane(idx, cam, blk_name, side_name, seed):
        blk = shm.Block(blk_name, [(n, tuple(s), d) for n, s, d in p["layout%d" % idx]])
        side = shm.Block(side_name, side_layout(p)); side_seq = side.seq; sd = None
        dr = mj.MjData(m); Rd = mj.Renderer(m, p["DH"], p["DW"]); Rd.enable_depth_rendering()
        depth = depth_model(mj, m, Rd, p, seed); rj = np.random.default_rng(seed + 100)
        seen = pose.seq; po = None; drops = 0
        ready.wait()
        while True:
            if not pose.wait_new(seen, 1.0):
                if not os.path.exists("/proc/%d" % world_pid): os._exit(0)
                continue
            s, st_ns, po = pose.read(po)
            t_pick = time.monotonic_ns()
            if seen and s - seen > 2: drops += (s - seen) // 2 - 1          # seqlock: +2 per pose write
            seen = s
            dr.qpos[:] = po["qpos"]; dr.mocap_pos[:] = po["mpos"][:m.nmocap]; dr.mocap_quat[:] = po["mquat"][:m.nmocap]
            mj.mj_kinematics(m, dr); mj.mj_camlight(m, dr)
            d = depth(dr, cam)
            if po["blind"][idx - 1]: d[:] = 0
            t_noise = time.monotonic_ns()
            extra = {}
            if side.seq != side_seq and not (side.seq & 1):
                side_seq, _, sd = side.read(sd); extra = sd
            stamp = st_ns + (int(rj.normal(0, jit) * 1e9) if jit else 0)
            t_hold = time.monotonic_ns(); wait = st_ns + lat_ns - t_hold
            if wait > 0: time.sleep(wait * 1e-9)
            t_pub = time.monotonic_ns(); blk.write(stamp, depth=d, **extra); t_w = time.monotonic_ns()
            if log is not None:
                with llock: log.write("%d %d %d %d %d %d %d %d %d %d\n" % (idx, st_ns, stamp, t_pick, depth.t_render[0], t_noise, t_pub, t_w, int(bool(extra)), drops))

    for i, (cam, bn, sn) in enumerate((("rs1_depth", "kevin_rs1", "kevin_rs1_side"), ("rs2_ir1", "kevin_rs2", "kevin_rs2_side")), 1):
        threading.Thread(target=lane, args=(i, cam, bn, sn, 7 + 13 * i), daemon=True, name="depth_rs%d" % i).start()
    ready.wait()
    while os.path.exists("/proc/%d" % world_pid): time.sleep(0.5)


if __name__ == "__main__":
    main(sys.argv[1])
