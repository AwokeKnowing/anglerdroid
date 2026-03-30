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

# ── Shaders for GPU depth-forward processing ─────────────────────

_VERT_SCATTER_OBS = """
#version 330
uniform mat3  u_rot;
uniform vec3  u_pivot;
uniform vec3  u_trans;
uniform float u_scale;
uniform vec2  u_offset;
uniform float u_cam_h;
uniform float u_sin_p;
uniform float u_cos_p;
uniform float u_floor;
uniform float u_ceil;
uniform float u_y_off;
uniform vec2  u_fbo_sz;
in vec3 in_v;
flat out float v_h;
void main() {
    gl_PointSize = 1.0;
    v_h = 0.0;
    vec3 p = in_v;
    if (p.z <= 0.0) { gl_Position = vec4(2.0,2.0,0.0,1.0); return; }
    p.y += u_y_off;
    float phys_h = u_cam_h - p.y * u_cos_p - p.z * u_sin_p;
    vec3 r = u_rot * (p - u_pivot) + u_pivot - u_trans;
    if (r.z <= u_floor || r.z >= u_ceil) {
        gl_Position = vec4(2.0,2.0,0.0,1.0); return;
    }
    float h_cm = clamp(phys_h * 100.0, 1.0, 100.0);
    vec2 px = r.xy * u_scale + u_offset;
    vec2 ndc = px / u_fbo_sz * 2.0 - 1.0;
    gl_Position = vec4(ndc, h_cm / 50.0 - 1.0, 1.0);
    v_h = h_cm;
}
"""

_FRAG_SCATTER_OBS = """
#version 330
flat in float v_h;
out vec4 fc;
void main() { fc = vec4(v_h / 255.0, 0.0, 0.0, 1.0); }
"""

_VERT_SCATTER_RANGE = """
#version 330
uniform mat3  u_rot;
uniform vec3  u_pivot;
uniform vec3  u_trans;
uniform float u_scale;
uniform vec2  u_offset;
uniform float u_y_off;
uniform vec2  u_cam_px;
uniform float u_max_r;
in vec3 in_v;
flat out float v_dist;
void main() {
    gl_PointSize = 1.0;
    v_dist = 0.0;
    vec3 p = in_v;
    if (p.z <= 0.0) { gl_Position = vec4(2.0,2.0,0.0,1.0); return; }
    p.y += u_y_off;
    vec3 r = u_rot * (p - u_pivot) + u_pivot - u_trans;
    vec2 px = r.xy * u_scale + u_offset;
    vec2 d = px - u_cam_px;
    float dist = length(d);
    if (dist < 0.5 || dist > u_max_r) {
        gl_Position = vec4(2.0,2.0,0.0,1.0); return;
    }
    float ndc_x = fract(atan(d.y, d.x) / 6.2831853 + 0.5) * 2.0 - 1.0;
    gl_Position = vec4(ndc_x, 0.0, dist / u_max_r * 2.0 - 1.0, 1.0);
    v_dist = dist;
}
"""

_FRAG_SCATTER_RANGE = """
#version 330
flat in float v_dist;
uniform float u_max_r;
out vec4 fc;
void main() { fc = vec4(v_dist / u_max_r, 0.0, 0.0, 1.0); }
"""

_VERT_FSQUAD = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_FRAG_RAYCAST = """
#version 330
uniform sampler2D u_obs;
uniform sampler2D u_range;
uniform vec2  u_cam;
uniform vec2  u_size;
uniform float u_max_r;
out vec4 fc;
void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 d = px - u_cam;
    float dist = length(d);
    if (dist < 0.5) { fc = vec4(1.0); return; }
    float angle = fract(atan(d.y, d.x) / 6.2831853 + 0.5);
    float max_d = texture(u_range, vec2(angle, 0.5)).r * u_max_r;
    if (dist > max_d + 1.5) { fc = vec4(0.0); return; }
    vec2 dir = d / dist;
    vec2 inv_sz = 1.0 / u_size;
    for (float t = 1.0; t < dist; t += 1.0) {
        vec2 s = (u_cam + dir * t) * inv_sz;
        if (s.x >= 0.0 && s.x < 1.0 && s.y >= 0.0 && s.y < 1.0) {
            if (texture(u_obs, s).r > 0.002) { fc = vec4(0.0); return; }
        }
    }
    fc = vec4(1.0);
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

    # ── Depth-forward configuration (call once before use) ───────

    def configure_depth_forward(self, rotation, pivot, translation,
                                 px_size, cam_height, sin_pitch, cos_pitch,
                                 floor_clip, height_clip,
                                 out_h, out_w, n_rays, max_ray_r):
        self._df_rot = rotation.astype(np.float32)
        self._df_pivot = pivot.astype(np.float32)
        self._df_trans = translation.astype(np.float32)
        self._df_scale = float(1.0 / px_size)
        self._df_offset = np.float32([out_w / 2.0,
                                      out_h / 2.0 + self._df_scale])
        self._df_cam_h = float(cam_height)
        self._df_sin_p = float(sin_pitch)
        self._df_cos_p = float(cos_pitch)
        self._df_floor = float(floor_clip)
        self._df_ceil = float(height_clip)
        self._df_out_h = out_h
        self._df_out_w = out_w
        self._df_n_rays = n_rays
        self._df_max_r = float(max_ray_r)

        cam_w = (np.dot(-self._df_pivot, self._df_rot)
                 + self._df_pivot - self._df_trans)
        self._df_cam_px = (cam_w[:2] * self._df_scale
                           + self._df_offset).astype(np.float32)
        self._df_morph_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (4, 4))
        self._df_configured = True
        self._df_gl_ready = False

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

    # ── Depth-forward GL init ────────────────────────────────────

    def _init_depth_gl(self):
        ctx = self._ctx
        ow, oh = self._df_out_w, self._df_out_h
        nr = self._df_n_rays

        self._df_obs_tex = ctx.texture((ow, oh), 1)
        self._df_obs_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._df_obs_depth = ctx.depth_renderbuffer((ow, oh))
        self._df_obs_fbo = ctx.framebuffer(
            color_attachments=[self._df_obs_tex],
            depth_attachment=self._df_obs_depth)

        self._df_range_tex = ctx.texture((nr, 1), 1)
        self._df_range_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._df_range_depth = ctx.depth_renderbuffer((nr, 1))
        self._df_range_fbo = ctx.framebuffer(
            color_attachments=[self._df_range_tex],
            depth_attachment=self._df_range_depth)

        self._df_known_tex = ctx.texture((ow, oh), 1)
        self._df_known_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._df_known_fbo = ctx.framebuffer(
            color_attachments=[self._df_known_tex])

        self._df_obs_upload_tex = ctx.texture((ow, oh), 1)
        self._df_obs_upload_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        max_pts = 320 * 240
        self._df_vbo = ctx.buffer(reserve=max_pts * 12)

        self._df_prog_obs = ctx.program(
            vertex_shader=_VERT_SCATTER_OBS,
            fragment_shader=_FRAG_SCATTER_OBS)
        self._df_prog_range = ctx.program(
            vertex_shader=_VERT_SCATTER_RANGE,
            fragment_shader=_FRAG_SCATTER_RANGE)
        self._df_prog_rc = ctx.program(
            vertex_shader=_VERT_FSQUAD,
            fragment_shader=_FRAG_RAYCAST)

        self._df_vao_obs = ctx.vertex_array(
            self._df_prog_obs, [(self._df_vbo, '3f', 'in_v')])
        self._df_vao_range = ctx.vertex_array(
            self._df_prog_range, [(self._df_vbo, '3f', 'in_v')])

        fsq = np.float32([-1, -1, 1, -1, -1, 1, 1, 1])
        fsq_buf = ctx.buffer(fsq.tobytes())
        self._df_vao_rc = ctx.vertex_array(
            self._df_prog_rc, [(fsq_buf, '2f', 'in_pos')])

        rot, piv, trans = self._df_rot, self._df_pivot, self._df_trans

        p = self._df_prog_obs
        p['u_rot'].write(rot.tobytes())
        p['u_pivot'].value = tuple(piv.tolist())
        p['u_trans'].value = tuple(trans.tolist())
        p['u_scale'].value = self._df_scale
        p['u_offset'].value = tuple(self._df_offset.tolist())
        p['u_cam_h'].value = self._df_cam_h
        p['u_sin_p'].value = self._df_sin_p
        p['u_cos_p'].value = self._df_cos_p
        p['u_floor'].value = self._df_floor
        p['u_ceil'].value = self._df_ceil
        p['u_fbo_sz'].value = (float(ow), float(oh))

        p = self._df_prog_range
        p['u_rot'].write(rot.tobytes())
        p['u_pivot'].value = tuple(piv.tolist())
        p['u_trans'].value = tuple(trans.tolist())
        p['u_scale'].value = self._df_scale
        p['u_offset'].value = tuple(self._df_offset.tolist())
        p['u_cam_px'].value = tuple(self._df_cam_px.tolist())
        p['u_max_r'].value = self._df_max_r

        p = self._df_prog_rc
        p['u_obs'].value = 0
        p['u_range'].value = 1
        p['u_cam'].value = tuple(self._df_cam_px.tolist())
        p['u_size'].value = (float(ow), float(oh))
        p['u_max_r'].value = self._df_max_r

        try:
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        except Exception:
            pass

        self._df_gl_ready = True
        self._df_n = 0
        print("gpu_render: depth_forward ready %dx%d  %d rays  max_r=%d" % (
            ow, oh, nr, int(self._df_max_r)))

    # ── GPU depth-forward pipeline ────────────────────────────────

    def depth_forward_gpu(self, verts, y_offset=0.0):
        """Process RS2 forward depth fully on GPU: scatter + morph + raycast.
        Returns (obs, known) as (out_h, out_w) uint8, or None on failure."""
        if not self.available or not getattr(self, '_df_configured', False):
            return None
        if not self._gl_ready:
            try:
                self._init_gl()
                self._gl_ready = True
            except Exception as e:
                print("gpu_render: init failed: %s" % e)
                self.available = False
                return None
        if not getattr(self, '_df_gl_ready', False):
            try:
                self._init_depth_gl()
            except Exception as e:
                print("gpu_render: depth init failed: %s" % e)
                import traceback; traceback.print_exc()
                return None

        t0 = time.monotonic()
        oh, ow = self._df_out_h, self._df_out_w
        n_pts = min(len(verts), 320 * 240)

        self._df_vbo.write(
            np.ascontiguousarray(verts[:n_pts], dtype=np.float32).tobytes())
        self._df_prog_obs['u_y_off'].value = float(y_offset)
        self._df_prog_range['u_y_off'].value = float(y_offset)

        # Pass 1: scatter obstacles → obs FBO (max-height via depth test)
        self._df_obs_fbo.use()
        self._df_obs_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=0.0)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_func = '>'
        self._df_vao_obs.render(moderngl.POINTS, vertices=n_pts)

        # Pass 2: scatter max-distance per angle → range FBO
        self._df_range_fbo.use()
        self._df_range_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=0.0)
        self._df_vao_range.render(moderngl.POINTS, vertices=n_pts)
        t1 = time.monotonic()

        # Read back obs for CPU morph close (77 KB, fast)
        obs_data = self._df_obs_fbo.read(components=1, alignment=1)
        obs = np.frombuffer(obs_data, dtype=np.uint8).reshape(oh, ow).copy()

        obs_bin = (obs > 0).astype(np.uint8)
        cv2.morphologyEx(obs_bin, cv2.MORPH_CLOSE, self._df_morph_kern,
                         iterations=2, dst=obs_bin)
        obs[obs_bin & (obs == 0)] = 1
        t2 = time.monotonic()

        # Upload morph-closed obs and raycast → known FBO
        self._df_obs_upload_tex.write(obs.tobytes())
        self._df_known_fbo.use()
        self._df_known_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._df_obs_upload_tex.use(location=0)
        self._df_range_tex.use(location=1)
        self._df_vao_rc.render(moderngl.TRIANGLE_STRIP)
        t3 = time.monotonic()

        # Read back known mask
        known_data = self._df_known_fbo.read(components=1, alignment=1)
        known = np.frombuffer(
            known_data, dtype=np.uint8).reshape(oh, ow).copy()

        self._ctx.depth_func = '<'

        self._df_n += 1
        t4 = time.monotonic()
        if self._df_n <= 3 or self._df_n % 100 == 0:
            print("gpu_depth: scatter=%.1fms morph=%.1fms "
                  "raycast=%.1fms read=%.1fms total=%.1fms" % (
                      (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                      (t3 - t2) * 1e3, (t4 - t3) * 1e3,
                      (t4 - t0) * 1e3))

        return obs, known

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
        self._ctx.depth_func = '<'

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

        # CPU overlays
        if not hasattr(self, '_rn'):
            self._rn = 0
        self._rn += 1
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
