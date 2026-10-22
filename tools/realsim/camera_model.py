"""Renderer-agnostic camera/sensor model: linear scene-referred frames in -> 8-bit frames like the real camera out.

Order follows the physical pipeline: (motion) exposure integration over subframes [+ rolling-shutter row timing]
-> lens (distortion, vignetting, MTF blur) -> sensor (photon shot noise, read noise, gain, clipping)
-> ISP (AE with lag, AWB with lag, gamma/tone, JPEG/MJPG).  All parameters are a dict so each camera
(mast UVC, RS D435 RGB, RS IR) is one parameter set.
"""
import numpy as np, cv2

MAST = dict(  # mast RGB, stage-A casita fit + assumptions marked A:
    f=None, cx=None, cy=None, k1=-0.0262, k2=0.0,
    vignette=0.35,          # A: cos^4-like falloff strength at the image corner (not yet measured)
    mtf_sigma=0.7,          # A: lens+demosaic blur, px
    full_well_e=6000.0,     # A: small-pixel UVC sensor
    read_noise_e=3.0,       # A
    exp_cap_s=0.030,        # measured config: auto exposure capped at 30 ms
    ae_target=0.45,         # A: mean of display-referred luma the AE converges to (typ. UVC 0.35-0.5)
    ae_alpha=0.15,          # A: per-frame AE adaptation rate (lag)
    awb_strength=0.3,       # A: partial grey-world AWB (real UVC AWB leaves a warm cast indoors)
    awb_alpha=0.1,          # A: per-frame AWB adaptation rate
    jpeg_q=80,              # A: MJPG stream quality
    rolling_readout_s=0.025,  # A: row readout span; D435 RGB (OV2740) is rolling, D435 depth (OV9282) is global
)


def integrate_exposure(subframes, rows=None, readout_frac=0.0):
    """Average N linear subframes rendered at poses spread over the exposure -> motion blur.
    With readout_frac>0 (rolling shutter) each image row r averages a window of subframes shifted by
    r/H*readout_frac of the subframe list (render subframes spanning exposure+readout)."""
    S = np.stack(subframes, 0)
    if readout_frac <= 0:
        return S.mean(0)
    N, H = S.shape[0], S.shape[1]; n_exp = max(1, int(round(N * (1 - readout_frac))))
    out = np.empty_like(S[0])
    for r in range(H):
        s0 = int(round(r / (H - 1) * (N - n_exp))); out[r] = S[s0:s0 + n_exp, r].mean(0)
    return out


def distort(img_lin, src_K, dst, k1, k2=0.0):
    """Resample a pinhole render (larger margin canvas, intrinsics src_K=(f,cx,cy)) into the Brown-k1/k2 image."""
    f, cx, cy, W, H = dst; fs, cxs, cys = src_K
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xd, yd = (u - cx) / f, (v - cy) / f; xn, yn = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = xn * xn + yn * yn; s = 1 + k1 * r2 + k2 * r2 * r2; xn, yn = xd / s, yd / s
    return cv2.remap(img_lin, (fs * xn + cxs).astype(np.float32), (fs * yn + cys).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def veiling_glare(img, k, sigma_px):
    """cheap-lens veiling glare: a fraction k of the image light scattered with a wide Gaussian PSF (sigma in px);
    brightens the surroundings of lamps (ceiling) the way the real UVC lens does."""
    if k <= 0: return img
    return (1 - k) * img + k * cv2.GaussianBlur(img, (0, 0), sigma_px)


def vignette(img, strength, f, cx, cy):
    H, W = img.shape[:2]; u, v = np.meshgrid(np.arange(W), np.arange(H))
    th = np.arctan(np.hypot(u - cx, v - cy) / f); c4 = np.cos(th) ** 4
    return img * ((1 - strength) + strength * c4)[..., None]


def srgb(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class Sensor:
    def __init__(self, p, seed=0):
        self.p = p; self.rng = np.random.default_rng(seed); self.k_exp = None; self.wb = np.ones(3)

    def frame(self, lin, steady=False):
        p = self.p
        lum = (lin * [0.2126, 0.7152, 0.0722]).sum(-1); H, W = lum.shape
        w = np.exp(-(((np.arange(H)[:, None] - H / 2) / (0.35 * H)) ** 2 + ((np.arange(W)[None] - W / 2) / (0.35 * W)) ** 2))
        meas = (srgb(lum * (self.k_exp or 1.0)) * w).sum() / w.sum()
        # AE: multiplicative controller on the exposure scale, applied with lag
        want = (self.k_exp or 1.0) * (p["ae_target"] / max(meas, 1e-4)) ** 2.2
        if steady:
            for _ in range(30):
                m = (srgb(lum * want) * w).sum() / w.sum(); want *= (p["ae_target"] / max(m, 1e-4)) ** 2.2
            self.k_exp = want
        else:
            self.k_exp = want if self.k_exp is None else self.k_exp * (want / self.k_exp) ** p["ae_alpha"]
        x = lin * self.k_exp
        # AWB: partial grey-world gains with lag
        xs = x.reshape(-1, 3); ok = (xs < 0.98).all(1)   # AWB statistics exclude clipped pixels (lamps), as ISPs do
        mean = xs[ok].mean(0) if ok.any() else xs.mean(0); g = mean[1] / np.maximum(mean, 1e-6)
        g_t = 1 + p["awb_strength"] * (g - 1)
        self.wb = g_t if steady else self.wb + p["awb_alpha"] * (g_t - self.wb)
        x = x * self.wb * np.asarray(p.get("awb_bias", (1.0, 1.0, 1.0)))   # residual cast the real ISP leaves after AWB
        # sensor: photons -> electrons with shot + read noise, gain set so mid-scale ~ full-well/2
        e = np.clip(x, 0, None) * p["full_well_e"] * 0.5
        e = self.rng.poisson(e).astype(np.float32) + self.rng.normal(0, p["read_noise_e"], e.shape)
        x = np.clip(e / (p["full_well_e"] * 0.5), 0, 1)
        y = (srgb(x) * 255 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", y[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_q"]])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def mast_pipeline(lin_canvas, meta, seed=0, p=MAST):
    """lin_canvas: pinhole linear RGB with margin; meta from scene.py json."""
    off_x, off_y = (meta["RW"] - meta["W"]) / 2, (meta["RH"] - meta["H"]) / 2
    img = distort(lin_canvas.astype(np.float32), (meta["f"], meta["cx"] + off_x, meta["cy"] + off_y),
                  (meta["f"], meta["cx"], meta["cy"], meta["W"], meta["H"]), meta["k1"], p.get("k2", 0.0))
    img = vignette(img, p["vignette"], meta["f"], meta["cx"], meta["cy"])
    img = cv2.GaussianBlur(img, (0, 0), p["mtf_sigma"])
    return Sensor(p, seed).frame(img, steady=True)


# ---- D4xx depth artefact model (applied to ground-truth depth from the renderer) ----
D435 = dict(f_px=None, baseline_m=0.050, subpix_rms=0.08,  # Intel tuning whitepaper: 0.07-0.11 with projector
            edge_flying_px=2, edge_thresh_m=0.08, dark_albedo=0.04, spec_thresh=0.6, sun_ir_thresh=0.9)


def d4xx_depth(z, f_px, p=D435, albedo=None, specular=None, sun_ir=None, seed=0):
    """z: GT depth (m), same size as the depth stream. Returns simulated D4xx depth (0 = hole).
    RMS error from the published model: dz = z^2 * subpix / (f * B)."""
    rng = np.random.default_rng(seed); z = z.astype(np.float32).copy()
    sig = z * z * p["subpix_rms"] / (f_px * p["baseline_m"])
    # disparity-domain noise is spatially correlated over the matching window (~5-7 px): blur the noise field
    n = cv2.GaussianBlur(rng.normal(0, 1, z.shape).astype(np.float32), (0, 0), 1.5); n /= n.std() + 1e-6
    out = z + sig * n
    # disparity quantisation (1/32 px subpixel steps)
    d = f_px * p["baseline_m"] / np.maximum(out, 1e-3); out = f_px * p["baseline_m"] / (np.round(d * 32) / 32 + 1e-6)
    # flying pixels on depth discontinuities: mix fg/bg along edges
    gz = np.maximum(np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)), np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))) / 8
    edge = cv2.dilate((gz > p["edge_thresh_m"]).astype(np.uint8), np.ones((3, 3)), iterations=p["edge_flying_px"]) > 0
    zmin = cv2.erode(z, np.ones((5, 5))); zmax = cv2.dilate(z, np.ones((5, 5))); t = rng.uniform(0, 1, z.shape)
    out = np.where(edge & (rng.uniform(0, 1, z.shape) < 0.5), zmin + t * (zmax - zmin), out)
    hole = (z <= 0.1) | (z > 10)
    if albedo is not None: hole |= albedo < p["dark_albedo"]             # IR-dark surfaces: no texture/return
    if specular is not None: hole |= specular > p["spec_thresh"]          # mirror-like: pattern reflects away
    if sun_ir is not None: hole |= sun_ir > p["sun_ir_thresh"]            # sunlit patches wash out the projector
    # left-camera occlusion shadow: holes on the left side of near objects (D4xx left imager is the reference)
    occl = (cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8) < -p["edge_thresh_m"]
    hole |= cv2.dilate(occl.astype(np.uint8), np.ones((1, 4))) > 0
    out[hole] = 0
    return out
