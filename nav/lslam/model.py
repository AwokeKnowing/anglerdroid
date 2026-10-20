"""SE(2) helpers, the odometry error model and the chi-square gate.  The only chosen number is ALPHA (false-reject rate);
the odometry and match error parameters are MEASURED (lslam_replay.py calib -> params json) and passed in."""
import math
import numpy as np
from scipy.stats import chi2

ALPHA = 0.01


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2]); return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])
def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2]); return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])
def between(a, b): return compose(inverse(a), b)                      # a^-1 o b
def gate(dof, alpha=ALPHA): return float(chi2.ppf(1 - alpha, dof))


class OdomModel:
    """Increment noise in the robot frame: sd_xy = a*dist + b, sd_th = c*|rot| + d*dist (+ e per second: gyro bias walk)."""
    def __init__(self, a, b, c, d, e=0.0): self.a, self.b, self.c, self.d, self.e = a, b, c, d, e

    def Q(self, inc, dt):
        dist, rot = math.hypot(inc[0], inc[1]), abs(inc[2])
        sxy = self.a * dist + self.b * (dist > 0 or rot > 0); sth = self.c * rot + self.d * dist + self.e * dt
        return np.diag([sxy ** 2, sxy ** 2, sth ** 2])

    def predict(self, x, P, inc, dt):
        """x <- x o inc; P <- J P J^T + G Q G^T (first order)."""
        c, s = math.cos(x[2]), math.sin(x[2])
        J = np.array([[1, 0, -s * inc[0] - c * inc[1]], [0, 1, c * inc[0] - s * inc[1]], [0, 0, 1.0]])
        G = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return compose(x, inc), J @ P @ J.T + G @ self.Q(inc, dt) @ G.T


def innovation(z, x):
    v = np.asarray(z, float) - np.asarray(x, float); v[2] = wrap(v[2]); return v


def nis(v, S): return float(v @ np.linalg.solve(S, v))


def fuse(x, P, z, R):
    """Kalman update with a direct pose measurement (H = I). Returns x, P, nis."""
    v = innovation(z, x); S = P + R; K = P @ np.linalg.inv(S)
    x2 = np.asarray(x, float) + K @ v; x2[2] = wrap(x2[2]); return x2, (np.eye(3) - K) @ P, nis(v, S)
