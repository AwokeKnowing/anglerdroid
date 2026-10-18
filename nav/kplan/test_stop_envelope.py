"""Planner-owned clearance: every executed rollout state can stop with gap (core.stop_reach) before the map.
python nav/kplan/test_stop_envelope.py   (CUDA; sets KEVIN_PLAN_OWNS_CLEARANCE=1)"""
import math, os, sys
os.environ['KEVIN_PLAN_OWNS_CLEARANCE'] = '1'          # the variant is behind this flag (default OFF)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import planner as KP, mppi_cuda as MC
from core import Limits as L, stop_reach, SIL12

FRONT = max(p[0] for p in SIL12)

def run(v0, a, w0=0.0, al=0.0, wall_x=3.0, H=60):
    N = 600; occ = np.zeros((N, N), bool); occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[:, int(wall_x * 100):] = True
    W = KP.World(occ, dev='cuda')
    U = torch.zeros(1, 1, H, 2, device='cuda'); U[..., 0] = a; U[..., 1] = al
    st = torch.tensor([[1.0, 3.0, 0.0, v0, w0]], device='cuda'); goal = torch.tensor([[5.5, 3.0, 0.0]], device='cuda')
    fp = KP.footprint_samples(0.03, 'cuda')
    _, dead, tx = MC.rollout(U, st, goal, fp, W.edt, L, 0.05, 6, erev=W.edt_rev)
    return tx[0, 0].cpu().numpy(), float(dead[0, 0])

def test_straight_at_wall():
    for v0 in (0.0, 0.3, 0.6):
        tx, dead = run(v0, L.a_max)
        assert dead == 0, (v0, 'rollout touched the wall')
        for x, y, th, v, w, _ in tx:
            s, _, g = stop_reach(v, w)
            gap_now = 3.0 - 0.005 - (x + FRONT)          # wall face = cell edge at 3.0 m; EDT measures to cell centres
            assert gap_now - s >= L.d_safe + g - 0.02, (v0, x, v, gap_now, s, g)   # 2 cm: 1 cm grid + 1 step of motion
        print('v0=%.1f: closest %.3f m, final v %.3f, required at 0.6 m/s %.3f m' % (v0, 3.0 - (tx[-1, 0] + FRONT), tx[-1, 3], L.d_safe + 2 * stop_reach(0.6, 0)[2]))

def test_free_space_full_speed():
    tx, dead = run(0.0, L.a_max, wall_x=5.9, H=40)
    assert dead == 0 and tx[-1, 3] > 0.59, tx[-1, 3]

def test_constants_measured():
    assert abs(L.a_brake - 3.0 * 2 * math.pi * 0.08565) < 0.02 and L.t_react == 0.15 and abs(L.d_safe - 0.045) < 1e-9
    s, th, g = stop_reach(0.6, 0.0); assert abs(s - (0.6 * 0.15 + 0.36 / (2 * L.a_brake))) < 1e-9 and th == 0

if __name__ == '__main__':
    test_constants_measured(); test_free_space_full_speed(); test_straight_at_wall(); print('test_stop_envelope ok')
