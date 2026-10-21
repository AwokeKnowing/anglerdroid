"""Spoke-ripple check on a KEVIN_RECORD log (real or sim): RS2 IMU at full rate + RS1 top-down floor depth per frame.
python -m emu.tools.scbench.ripple_real <record dir> [n_spokes=20] [wheel_r_m=twin]
Uses the moving part of the log (|v| > 0.05 m/s, |w| < 0.15 rad/s).  Spoke frequency f = n * v / (2 pi R).
RS1: median depth (mm) of the central 20 % of the image per frame (top-down floor; when the core owns RS1, the
recorded floor-points mean rs1_zc), its sd / peak-to-peak and the 30 Hz
spectrum (f above 15 Hz shows aliased at |f - 30|).  IMU: every sample nav drained (imu_g / imu_a / imu_ts, added
2026-10-21), consecutive duplicates removed, resampled to 400 Hz; amplitude spectrum of accel x/y/z and gyro x/y/z.
Reports the peak in 3-60 Hz and the amplitude at f +- 10 %.  Old logs without imu_* fall back to the one-per-tick sample."""
import glob, json, math, os, sys
import numpy as np

def spec(t, x, fs):
    tu = np.arange(t[0], t[-1], 1.0 / fs); y = np.interp(tu, t, x); y = y - y.mean()
    w = np.hanning(len(y)); A = np.abs(np.fft.rfft(y * w)) * 2 / w.sum(); f = np.fft.rfftfreq(len(y), 1.0 / fs)
    return f, A

def band(f, A, lo, hi):
    m = (f >= lo) & (f <= hi)
    if not m.any(): return None
    k = int(np.argmax(np.where(m, A, -1))); return round(float(f[k]), 2), float(A[k])

def main():
    d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if len(sys.argv) > 3: R = float(sys.argv[3])
    else:
        sys.path[:0] = [os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')]; import twin_geometry as TG; R = TG.WHEEL_R_M
    fs = sorted(glob.glob(os.path.join(d, 't*.npz')))
    T, V, W, Z, G, A, TS = [], [], [], [], [], [], []
    for p in fs:
        z = np.load(p, allow_pickle=False); m = json.loads(str(z['meta']))
        v, w = float(m.get('v') or 0), float(m.get('w') or 0)
        if 'vl' in m and m['vl'] is not None: v = 0.5 * (float(m['vl']) + float(m['vr']))
        if abs(v) < 0.05 or abs(w) > 0.15: continue
        T.append(float(m['t'])); V.append(v); W.append(w)
        if 'rs1' in z.files:
            r = z['rs1'].astype(float); h, wd = r.shape[:2]; c = r[int(h * .4):int(h * .6), int(wd * .4):int(wd * .6)]; c = c[c > 0]
            Z.append(float(np.median(c)) if c.size else np.nan)
        elif m.get('rs1_zc') is not None: Z.append(1000.0 * float(m['rs1_zc']))      # core-owned RS1: floor verts mean (nav/run.py)
        else: Z.append(np.nan)
        if 'imu_ts' in z.files: G.append(z['imu_g']); A.append(z['imu_a']); TS.append(z['imu_ts'])
        elif 'imu_t' in m and m['imu_t'] is not None and 'accel' in z.files: G.append(z['gyro'][None]); A.append(z['accel'][None]); TS.append(np.array([float(m['imu_t'])]))
    out = dict(dir=d, files=len(fs), moving_ticks=len(T))
    if len(T) < 30: out['error'] = 'fewer than 30 moving straight ticks'; print(json.dumps(out)); return
    T = np.array(T); v = float(np.median(V)); f0 = n * abs(v) / (2 * math.pi * R)
    out.update(v_med=round(v, 3), wheel_r=R, spoke_hz=round(f0, 2), rs1_alias_hz=round(abs(f0 - 30 * round(f0 / 30)), 2), dur_s=round(T[-1] - T[0], 1),
               tick_hz=round((len(T) - 1) / (T[-1] - T[0]), 1))
    Zf = np.array(Z); ok = np.isfinite(Zf)
    if ok.sum() > 30:
        zz = Zf[ok]; tz = T[ok]; zd = zz - np.convolve(zz, np.ones(31) / 31, 'same')    # remove floor slope / slow pitch
        zd = zd[15:-15] if len(zd) > 60 else zd
        f, S = spec(tz, zz, 30.0)
        out['rs1_floor_mm'] = dict(median=round(float(np.median(zz)), 1), sd_hp=round(float(zd.std()), 2), pp_hp=round(float(np.ptp(zd)), 2),
                                   peak=band(f, S, 1, 15), at_alias=band(f, S, 0.9 * out['rs1_alias_hz'], 1.1 * out['rs1_alias_hz'] + 0.2))
    if TS:
        g = np.concatenate(G).astype(float); a = np.concatenate(A).astype(float); ts = np.concatenate(TS).astype(float)
        o = np.argsort(ts, kind='stable'); g, a, ts = g[o], a[o], ts[o]
        keep = np.r_[True, np.any(np.diff(a, axis=0) != 0, axis=1) | np.any(np.diff(g, axis=0) != 0, axis=1)] & np.r_[True, np.diff(ts) > 0]
        g, a, ts = g[keep], a[keep], ts[keep]; rate = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
        out['imu'] = dict(samples=len(ts), rate_hz=round(rate, 1))
        if rate > 100:
            for k, x in (('ax', a[:, 0]), ('ay', a[:, 1]), ('az', a[:, 2]), ('gx', g[:, 0]), ('gy', g[:, 1]), ('gz', g[:, 2])):
                f, S = spec(ts, x, 400.0); fb = band(f, S, 0.9 * f0, 1.1 * f0)
                out['imu'][k] = dict(sd=round(float(x.std()), 4), peak_3_60=band(f, S, 3, 60), at_spoke=fb)
        else: out['imu']['note'] = 'one sample per tick only (log predates imu_g/imu_a/imu_ts): no full-rate spectrum'
    print(json.dumps(out, indent=1))

if __name__ == '__main__': main()
