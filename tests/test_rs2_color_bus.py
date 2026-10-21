"""RS2 colour over the bus (2026-10-21 bug: cameras.RSCamera.color is FRAME_H x FRAME_W (240x320) but the sc_rs2img slot
was 480x640, so Block.write packed it flat into the first rows and every reader got a scrambled image).
python tests/test_rs2_color_bus.py"""
import os, sys, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ["KEVIN_SC_PREFIX"] = "kevin_test_rs2c_%d" % os.getpid()            # private shm blocks
from nav import bus, camproc
import robot_config as RC


def test_slot_matches_producer():
    assert bus.RS2_COLOR_SHAPE == (RC.FRAME_H, RC.FRAME_W, 3), (bus.RS2_COLOR_SHAPE, RC.FRAME_H, RC.FRAME_W)
    src = open(os.path.join(ROOT, "src", "cameras.py")).read()
    assert "self.color = np.zeros((FRAME_H, FRAME_W, 3)" in src           # the producer's buffer


def test_roundtrip_and_reader():
    B = bus.open_all((480, 848), create=True)
    try:
        B["rs2info"].write(f=np.array([400.0, 400.0, 424.0, 240.0, 848, 480]), set=1)
        rd = camproc.Rs2Bus(B)
        img = np.zeros(bus.RS2_COLOR_SHAPE, np.uint8)
        img[..., 0] = np.arange(RC.FRAME_W)[None, :] % 256; img[..., 1] = np.arange(RC.FRAME_H)[:, None]; img[..., 2] = 77
        B["rs2img"].write(depth=np.full((480, 848), 1000, np.uint16), color=img, stamp=1.0, t_recv=1.0, n=1, has_ir=0)
        assert rd.wait(1.0)
        assert rd.color.shape == img.shape and np.array_equal(rd.color, img), "colour scrambled on the bus"
        assert rd.depth.shape == (480, 848) and int(rd.depth[100, 100]) == 1000
    finally:
        for b in B.values():
            try: b.shm.close(); b.shm.unlink()
            except Exception: pass


if __name__ == "__main__":
    test_slot_matches_producer(); print("PASS test_slot_matches_producer")
    test_roundtrip_and_reader(); print("PASS test_roundtrip_and_reader")
