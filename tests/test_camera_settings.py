"""cameras.apply_exposure applies the resolved cam_* settings (2026-10-21 exposure fix) -- fake RealSense sensors.
python tests/test_camera_settings.py"""
import os, sys, subprocess, types
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

PROBE = r'''
import sys, types, json; sys.path[:0] = [%r, %r]
import cameras, robot_params as RP
O = types.SimpleNamespace(**{k: k for k in ("enable_auto_exposure", "exposure", "gain", "auto_exposure_limit", "auto_exposure_limit_toggle",
     "auto_exposure_priority", "enable_auto_white_balance", "white_balance", "frames_queue_size")})
class ROI: pass
class S:
    def __init__(s, name): s.name, s.opts, s.roi = name, {}, None
    def supports(s, k): return True
    def set_option(s, k, v): s.opts[k] = v
    def get_info(s, k): return s.name
    def as_roi_sensor(s): return s
    def set_region_of_interest(s, r): s.roi = (r.min_x, r.min_y, r.max_x + 1, r.max_y + 1)
d, c = S("Stereo Module"), S("RGB Camera")
dev = types.SimpleNamespace(first_depth_sensor=lambda: d, sensors=[d, c])
cameras.rs = types.SimpleNamespace(option=O, camera_info=types.SimpleNamespace(name="name"), region_of_interest=ROI)
cameras.apply_exposure(types.SimpleNamespace(get_device=lambda: dev), RP.V["cam_rs1_serial"])
print(json.dumps(dict(depth=d.opts, rgb=c.opts, roi=c.roi, mode=RP.V["cam_exposure_mode"])))
''' % (ROOT, os.path.join(ROOT, "src"))


def probe(**env):
    e = dict(os.environ, **env); e.pop("KEVIN_ROBOT_PARAMS_NOLOAD", None)
    out = subprocess.run([sys.executable, "-c", PROBE], env=e, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    import json; return json.loads(out.stdout.strip().splitlines()[-1])


def test_auto_capped():
    r = probe(KEVIN_CAM_EXPOSURE="auto_capped")
    assert r["depth"]["enable_auto_exposure"] == 1 and r["depth"]["auto_exposure_limit"] == 30000 and r["depth"]["auto_exposure_limit_toggle"] == 1
    assert r["depth"]["auto_exposure_limit"] < 33333                       # can never stretch the 30 Hz frame period
    assert r["rgb"]["enable_auto_exposure"] == 1 and r["rgb"]["auto_exposure_priority"] == 0
    assert r["rgb"]["enable_auto_white_balance"] == 0 and r["rgb"]["white_balance"] == 4800
    assert tuple(r["roi"]) == (363, 222, 389, 252)                            # RS1 colour AE on the card


def test_legacy():
    r = probe(KEVIN_CAM_EXPOSURE="legacy")
    assert r["depth"] == {"enable_auto_exposure": 0, "exposure": 15000, "gain": 64}, r["depth"]


if __name__ == "__main__":
    test_auto_capped(); print("PASS test_auto_capped"); test_legacy(); print("PASS test_legacy")
