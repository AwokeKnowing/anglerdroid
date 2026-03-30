"""gpu_render.py – GPU-accelerated 3D terrain renderer (ModernGL + EGL).

Renders the global occupancy/height map as a displacement-mapped mesh
with a follow-cam, robot mesh, and CPU minimap overlay.
Designed for headless Jetson Orin NX operation.

Falls back gracefully if moderngl/EGL is unavailable — the caller
should check .available and use the software renderer in globalmap.py.
"""

import math
import time
import numpy as np
import cv2

try:
    import moderngl
    _HAS_MGL = True
except ImportError:
    _HAS_MGL = False

# ── Configuration ────────────────────────────────────────────────
GRID_DIV = 8           # map pixels per terrain grid vertex

CAM_BEHIND = 2.5       # m behind robot
CAM_HEIGHT = 3.0       # m above ground
CAM_LOOK_AHEAD = 1.0   # m ahead of robot (look-at target)
CAM_FOV_DEG = 60       # vertical FOV
FOG_DIST = 12.0
NEAR_CLIP = 0.1
FAR_CLIP = 50.0

MINIMAP_SZ = 200
MINIMAP_PAD = 8
MINIMAP_ZOOM = 0.25

PX_SIZE = 0.02
FREE_THRESH = 190
OBS_THRESH = 90
UNK_DISPLAY = 160

# ── Shaders ──────────────────────────────────────────────────────

_VERT_TERRAIN = """
#version 330

uniform mat4 u_mvp;
uniform sampler2D u_hmap;
uniform vec2 u_mapsz;
uniform vec2 u_origin;
uniform float u_px;
uniform float u_hscale;
uniform float u_hmin;

in vec2 in_uv;

out vec2 v_uv;
out float v_hcm;
out vec3 v_w;

void main() {
    v_uv = in_uv;
    float raw = texture(u_hmap, in_uv).r * 255.0;
    v_hcm = raw > 0.5 ? max(raw, u_hmin) : 0.0;
    float hm = v_hcm * u_hscale;

    vec3 w = vec3(
        (in_uv.x * u_mapsz.x - u_origin.x) * u_px,
        hm,
        (in_uv.y * u_mapsz.y - u_origin.y) * u_px);
    v_w = w;
    gl_Position = u_mvp * vec4(w, 1.0);
}
"""

_FRAG_TERRAIN = """
#version 330

uniform sampler2D u_conf;
uniform vec3 u_cam;
uniform float u_fogfar;
uniform int u_topdown;

in vec2 v_uv;
in float v_hcm;
in vec3 v_w;

out vec4 fc;

void main() {
    float c = texture(u_conf, v_uv).r * 255.0;
    vec3 col;

    if (u_topdown == 1) {
        if (c > 190.0) {
            col = vec3(1.0);
        } else if (c < 90.0) {
            float i = clamp(v_hcm / 100.0, 0.05, 1.0);
            col = vec3(i * 0.9, i * 0.75, i * 0.5);
        } else {
            col = vec3(0.627);
        }
        fc = vec4(col, 1.0);
        return;
    }

    if (c > 190.0) {
        col = vec3(0.765, 0.792, 0.725);
    } else if (c < 90.0) {
        float h = v_hcm;
        col = vec3(
            clamp((90.0 - h * 0.5) / 255.0, 0.12, 0.35),
            clamp((75.0 - h * 0.3) / 255.0, 0.10, 0.29),
            clamp((65.0 - h * 0.2) / 255.0, 0.08, 0.25));
    } else {
        col = vec3(0.569);
    }

    vec3 n = normalize(cross(dFdx(v_w), dFdy(v_w)));
    vec3 ld = normalize(vec3(0.3, -0.8, -0.5));
    float nl = max(dot(n, -ld), 0.0);
    col *= (0.4 + 0.6 * nl);

    float d = length(v_w - u_cam);
    float fog = clamp(d / u_fogfar, 0.0, 0.75);
    col = mix(col, vec3(0.569, 0.569, 0.588), fog);

    fc = vec4(col, 1.0);
}
"""

_VERT_ROBOT = """
#version 330

uniform mat4 u_mvp;
uniform mat3 u_nmat;

in vec3 in_pos;
in vec3 in_col;
in vec3 in_nrm;

out vec3 v_col;
out vec3 v_nrm;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_col = in_col;
    v_nrm = u_nmat * in_nrm;
}
"""

_FRAG_ROBOT = """
#version 330

in vec3 v_col;
in vec3 v_nrm;

out vec4 fc;

void main() {
    vec3 ld = normalize(vec3(0.3, -0.8, -0.5));
    float nl = max(dot(normalize(v_nrm), -ld), 0.0);
    fc = vec4(v_col * (0.35 + 0.65 * nl), 1.0);
}
"""


# ── Matrix helpers ───────────────────────────────────────────────

def _look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    rn = np.linalg.norm(r)
    if rn < 1e-6:
        r = np.float32([1, 0, 0])
    else:
        r = r / rn
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -r.dot(eye)
    m[1, 3] = -u.dot(eye)
    m[2, 3] = f.dot(eye)
    return m


def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _ortho(left, right, bottom, top, near, far):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


# ── Robot mesh triangulation ─────────────────────────────────────

def _triangulate_robot():
    """Build triangulated robot mesh VBO data. Returns float32 (N, 9)."""
    from globalmap import GlobalMap
    verts, faces, colors = GlobalMap._build_robot_mesh()

    data = []
    for i, fidx in enumerate(faces):
        v = verts[fidx]
        c = colors[i].astype(np.float32) / 255.0
        e1 = v[1] - v[0]
        e2 = v[-1] - v[0]
        n = np.cross(e1, e2)
        nl = np.linalg.norm(n)
        n = n / nl if nl > 1e-8 else np.float32([0, 0, 1])
        for j in range(1, len(fidx) - 1):
            for vi in (0, j, j + 1):
                data.append(np.concatenate([v[vi], c, n]))

    if not data:
        return np.zeros((0, 9), dtype=np.float32)
    return np.array(data, dtype=np.float32)


# ── Main renderer ────────────────────────────────────────────────

class GPURenderer:
    """GPU-accelerated 3D terrain view. Check .available after first render()."""

    def __init__(self, map_w, map_h, view_w, view_h):
        self.available = _HAS_MGL
        self._mw = map_w
        self._mh = map_h
        self._vw = view_w
        self._vh = view_h
        self.topdown = False
        self._gl_ready = False

        if not _HAS_MGL:
            print("gpu_render: moderngl not installed")

    # ── GL init ──────────────────────────────────────────────────

    def _init_gl(self):
        t0 = time.monotonic()

        try:
            self._ctx = moderngl.create_context(
                standalone=True, backend='egl')
        except Exception:
            self._ctx = moderngl.create_context(standalone=True)

        # Framebuffer (texture-backed for reliable readback)
        self._color_tex_fbo = self._ctx.texture(
            (self._vw, self._vh), 4)
        depth = self._ctx.depth_renderbuffer((self._vw, self._vh))
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._color_tex_fbo],
            depth_attachment=depth)

        # Map textures (preallocated, updated per-frame)
        self._tex_conf = self._ctx.texture((self._mw, self._mh), 1)
        self._tex_hmap = self._ctx.texture((self._mw, self._mh), 1)
        self._tex_conf.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._tex_hmap.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Terrain shader
        self._prog_t = self._ctx.program(
            vertex_shader=_VERT_TERRAIN, fragment_shader=_FRAG_TERRAIN)
        self._prog_t['u_mapsz'].value = (float(self._mw), float(self._mh))
        self._prog_t['u_origin'].value = (
            float(self._mw // 2), float(self._mh // 2))
        self._prog_t['u_px'].value = PX_SIZE
        self._prog_t['u_hscale'].value = 0.01
        self._prog_t['u_hmin'].value = 5.0
        self._prog_t['u_fogfar'].value = FOG_DIST
        self._prog_t['u_hmap'].value = 0
        self._prog_t['u_conf'].value = 1
        self._prog_t['u_topdown'].value = 0

        self._build_terrain()

        # Robot shader + mesh
        self._prog_r = self._ctx.program(
            vertex_shader=_VERT_ROBOT, fragment_shader=_FRAG_ROBOT)
        self._build_robot()

        # Pre-allocated readback buffer
        self._out = np.empty((self._vh, self._vw, 3), dtype=np.uint8)

        # PBO double-buffer for async readback
        pbo_sz = self._vw * self._vh * 3
        self._pbo = [self._ctx.buffer(reserve=pbo_sz),
                     self._ctx.buffer(reserve=pbo_sz)]
        self._pbo_idx = 0
        self._pbo_ready = False

        t1 = time.monotonic()
        print("gpu_render: ready %dx%d  grid=%dx%d  robot=%d tris  %.0fms" % (
            self._vw, self._vh, self._gw, self._gh,
            self._robot_ntris, (t1 - t0) * 1e3))

    def _build_terrain(self):
        gw = self._mw // GRID_DIV
        gh = self._mh // GRID_DIV
        self._gw, self._gh = gw, gh

        u = np.linspace(0, 1, gw + 1, dtype=np.float32)
        v = np.linspace(0, 1, gh + 1, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        verts = np.column_stack([uu.ravel(), vv.ravel()])

        stride = gw + 1
        rows = np.arange(gh, dtype=np.int32)[:, None]
        cols = np.arange(gw, dtype=np.int32)[None, :]
        tl = (rows * stride + cols).ravel()
        tr = tl + 1
        bl = tl + stride
        br = bl + 1
        idx = np.column_stack([tl, bl, tr, tr, bl, br]).ravel()

        vbo = self._ctx.buffer(verts.tobytes())
        ibo = self._ctx.buffer(idx.tobytes())
        self._vao_t = self._ctx.vertex_array(
            self._prog_t, [(vbo, '2f', 'in_uv')],
            index_buffer=ibo, index_element_size=4)

    def _build_robot(self):
        data = _triangulate_robot()
        self._robot_ntris = len(data) // 3
        if len(data) == 0:
            self._vao_r = None
            return
        vbo = self._ctx.buffer(data.tobytes())
        self._vao_r = self._ctx.vertex_array(
            self._prog_r,
            [(vbo, '3f 3f 3f', 'in_pos', 'in_col', 'in_nrm')])

    # ── Per-frame render ─────────────────────────────────────────

    def render(self, x, y, theta, conf_map, height_map,
               trail_xy=None, fwd_scale=1.0, bwd_scale=1.0, ang_scale=1.0):
        """Render full 3D view. Returns (vh, vw, 3) uint8 RGB array."""
        if not self._gl_ready:
            try:
                self._init_gl()
                self._gl_ready = True
            except Exception as e:
                print("gpu_render: init failed: %s" % e)
                import traceback
                traceback.print_exc()
                self.available = False
                return None

        t0 = time.monotonic()

        # Collect previous frame's async readback (DMA ran during inter-frame gap)
        if self._pbo_ready:
            prev_pbo = self._pbo[1 - self._pbo_idx]
            data = prev_pbo.read()
            self._out[:] = np.frombuffer(data, dtype=np.uint8).reshape(
                self._vh, self._vw, 3)[::-1]

        # Upload textures
        self._tex_conf.write(conf_map.tobytes())
        self._tex_hmap.write(height_map.tobytes())

        ct, st = math.cos(theta), math.sin(theta)

        # Camera: GL coords (X=east, Y=up, Z = -world_y)
        if self.topdown:
            cam = np.float32([x, 15.0, -y])
            tgt = np.float32([x, 0.0, -y])
            up = np.float32([ct, 0, -st])
            half_ext = 6.0
            aspect = self._vw / self._vh
            view = _look_at(cam, tgt, up)
            proj = _ortho(-half_ext * aspect, half_ext * aspect,
                          -half_ext, half_ext, 0.1, 100.0)
            self._prog_t['u_topdown'].value = 1
        else:
            cam = np.float32([
                x - CAM_BEHIND * ct,
                CAM_HEIGHT,
                -(y - CAM_BEHIND * st)])
            tgt = np.float32([
                x + CAM_LOOK_AHEAD * ct,
                0.0,
                -(y + CAM_LOOK_AHEAD * st)])
            up = np.float32([0, 1, 0])
            view = _look_at(cam, tgt, up)
            proj = _perspective(CAM_FOV_DEG,
                                self._vw / self._vh, NEAR_CLIP, FAR_CLIP)
            self._prog_t['u_topdown'].value = 0

        mvp = proj @ view

        # Robot model: local +X fwd, +Y left, +Z up → GL
        R0 = np.float32([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        Ry = np.float32([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        R_robot = Ry @ R0
        model = np.eye(4, dtype=np.float32)
        model[:3, :3] = R_robot
        model[:3, 3] = [x, 0, -y]
        mvp_robot = proj @ view @ model

        # ── Draw ──
        self._fbo.use()
        self._fbo.clear(0.569, 0.569, 0.588, 1.0)
        self._ctx.enable(moderngl.DEPTH_TEST)

        # Terrain
        self._tex_hmap.use(location=0)
        self._tex_conf.use(location=1)
        self._prog_t['u_mvp'].write(mvp.T.astype(np.float32).tobytes())
        self._prog_t['u_cam'].value = tuple(cam.tolist())
        self._vao_t.render()

        # Robot
        if self._vao_r is not None:
            self._prog_r['u_mvp'].write(
                mvp_robot.T.astype(np.float32).tobytes())
            self._prog_r['u_nmat'].write(
                R_robot.T.astype(np.float32).tobytes())
            self._vao_r.render()

        # Kick async PBO readback (returns immediately, DMA runs in background)
        cur_pbo = self._pbo[self._pbo_idx]
        self._fbo.read_into(cur_pbo, components=3, alignment=1)

        if not self._pbo_ready:
            # First frame only: must block synchronously
            data = cur_pbo.read()
            self._out[:] = np.frombuffer(data, dtype=np.uint8).reshape(
                self._vh, self._vw, 3)[::-1]
            self._pbo_ready = True

        self._pbo_idx = 1 - self._pbo_idx
        t1 = time.monotonic()

        # CPU overlays (minimap every 4th frame to save ~3ms)
        if not hasattr(self, '_rn'):
            self._rn = 0
        self._rn += 1
        if self._rn % 4 == 1:
            self._draw_minimap(self._out, x, y, theta, conf_map, height_map,
                               fwd_scale, bwd_scale, ang_scale)
            if trail_xy is not None and len(trail_xy) >= 2:
                self._draw_trail(self._out, trail_xy, view, proj)

        t2 = time.monotonic()
        if self._rn <= 3 or self._rn % 100 == 0:
            print("gpu_render: gpu+read=%.1fms  overlay=%.1fms  total=%.1fms" % (
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3))

        return self._out

    # ── CPU minimap ──────────────────────────────────────────────

    def _draw_minimap(self, out, x, y, theta, conf, hmap,
                      fwd_scale, bwd_scale, ang_scale):
        rc = int(self._mw // 2 + x / PX_SIZE)
        rr = int(self._mh // 2 - y / PX_SIZE)

        M = np.float64([
            [MINIMAP_ZOOM, 0, MINIMAP_SZ / 2 - rc * MINIMAP_ZOOM],
            [0, MINIMAP_ZOOM, MINIMAP_SZ / 2 - rr * MINIMAP_ZOOM],
        ])
        mini_conf = cv2.warpAffine(
            conf, M, (MINIMAP_SZ, MINIMAP_SZ),
            flags=cv2.INTER_NEAREST, borderValue=128)

        rgb = np.full((MINIMAP_SZ, MINIMAP_SZ, 3), UNK_DISPLAY, dtype=np.uint8)
        rgb[mini_conf > FREE_THRESH] = [255, 255, 255]
        obs = mini_conf < OBS_THRESH
        if obs.any():
            mini_hm = cv2.warpAffine(
                hmap, M, (MINIMAP_SZ, MINIMAP_SZ),
                flags=cv2.INTER_NEAREST, borderValue=0)
            g = np.clip(64 - (mini_hm[obs].astype(np.int16) >> 1),
                        10, 64).astype(np.uint8)
            rgb[obs] = np.stack([g, g, g], axis=1)

        # Robot dot (color reflects safety throttle)
        sf = min(fwd_scale, bwd_scale, ang_scale)
        dot_col = (int(255 * (1 - sf)), int(200 * sf), int(255 * sf))
        cx, cy = MINIMAP_SZ // 2, MINIMAP_SZ // 2
        cv2.circle(rgb, (cx, cy), 3, dot_col, -1)

        # Direction tick
        dx = int(6 * math.cos(theta))
        dy = int(-6 * math.sin(theta))
        cv2.line(rgb, (cx, cy), (cx + dx, cy + dy), dot_col, 1)

        cv2.rectangle(rgb, (0, 0), (MINIMAP_SZ - 1, MINIMAP_SZ - 1),
                       (80, 80, 80), 1)

        y0, x0 = MINIMAP_PAD, MINIMAP_PAD
        y1, x1 = y0 + MINIMAP_SZ, x0 + MINIMAP_SZ
        if y1 <= out.shape[0] and x1 <= out.shape[1]:
            region = out[y0:y1, x0:x1]
            np.multiply(region, 0.15, out=region, casting='unsafe')
            np.add(region, (rgb * 0.85).astype(np.uint8),
                   out=region, casting='unsafe')

    # ── CPU trail ────────────────────────────────────────────────

    def _draw_trail(self, out, trail_xy, view, proj):
        n = len(trail_xy)
        pts_gl = np.empty((n, 4), dtype=np.float32)
        pts_gl[:, 0] = trail_xy[:, 0]
        pts_gl[:, 1] = 0.0
        pts_gl[:, 2] = -trail_xy[:, 1]
        pts_gl[:, 3] = 1.0

        mvp = proj @ view
        clip = (mvp @ pts_gl.T).T
        w = clip[:, 3]
        ok = w > 0.1
        if not ok.any():
            return

        ndc_x = clip[ok, 0] / w[ok]
        ndc_y = clip[ok, 1] / w[ok]
        sx = ((ndc_x + 1) * 0.5 * self._vw).astype(np.int32)
        sy = ((1 - ndc_y) * 0.5 * self._vh).astype(np.int32)

        keep = (sx >= 0) & (sx < self._vw) & (sy >= 0) & (sy < self._vh)
        pts = np.column_stack([sx[keep], sy[keep]])
        if len(pts) >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False,
                          (80, 140, 255), 2, cv2.LINE_AA)

    # ── Cleanup ──────────────────────────────────────────────────

    def release(self):
        if hasattr(self, '_ctx'):
            try:
                self._ctx.release()
            except Exception:
                pass
