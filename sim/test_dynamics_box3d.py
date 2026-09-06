"""Extra tests: dynamics fidelity + Box3D mast collide."""
from sim.dynamics import DiffDriveDynamics, V_MAX, WHEELBASE_M
from sim.box3d import house_boxes, mast_collision, body_collision, rasterize_boxes
from sim.robot import Robot


def test_dynamics_accel_limit():
    d = DiffDriveDynamics(latency_s=0.0)
    d.reset()
    v, w = d.apply(V_MAX, 0.0, 1.0 / 30.0)
    assert v < V_MAX * 0.9, v
    assert v > 0.0


def test_dynamics_latency():
    d = DiffDriveDynamics(latency_s=0.15, dt_nominal=1.0 / 30.0)
    d.reset()
    vs = []
    for _ in range(3):
        v, w = d.apply(0.2, 0.0, 1.0 / 30.0)
        vs.append(v)
    assert vs[0] == 0.0


def test_wheelbase_clip():
    d = DiffDriveDynamics(latency_s=0.0)
    v, w = d.clip_cmd(0.25, 2.0)
    assert abs(v) + abs(w) * (WHEELBASE_M * 0.5) <= V_MAX + 1e-6


def test_box3d_mast_hits_table_top():
    boxes = house_boxes()
    hit, name = mast_collision(boxes, 0.75, 1.9, 0.0)
    assert hit, name
    assert "table" in name


def test_box3d_open_floor_clear():
    boxes = house_boxes()
    hit, _ = mast_collision(boxes, 0.5, 0.5, 0.0)
    assert not hit
    hit2, _ = body_collision(boxes, 0.5, 0.5, 0.0)
    assert not hit2


def test_rasterize_shapes():
    obs, height = rasterize_boxes(house_boxes())
    assert obs.shape[0] > 100 and obs.shape[1] > 100
    assert height.max() >= 70


def test_robot_uses_dynamics():
    r = Robot(0.8, 1.2, 0.0)
    assert hasattr(r, "dyn")
    r.step(0.25, 0.0, 1.0 / 30.0)
    assert r.v < 0.25


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
            print("✅", k)
    print("All dynamics/box3d tests passed")
