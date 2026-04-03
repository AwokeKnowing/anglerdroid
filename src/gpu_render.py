"""gpu_render.py – GPU-accelerated 3D voxel terrain renderer (ModernGL + EGL).

Renders the full atlas: 3D voxel-column terrain with screen-space AO,
follow-cam, robot mesh, minimap, camera feeds, battery bar, and trail.
GPU-only pipeline — no CPU fallback.  Designed for Jetson Orin NX.
"""

import math
import time
import numpy as np

try:
    import moderngl
    _HAS_MGL = True
except ImportError:
    _HAS_MGL = False

# ── Configuration ────────────────────────────────────────────────
GRID_DIV = 1           # map pixels per mesh vertex (1 = 2cm mesh)
BLOB_DIV = 2           # map pixels per blob cell (2 = 4cm blur)
BLOB_SIGMA = 0.5       # Gaussian blur sigma in blob cells
BLOB_THRESH = 3.0      # height threshold in cm (removes blur tail)

CAM_BEHIND = 2.5       # m behind robot
CAM_HEIGHT = 3.0       # m above ground
CAM_LOOK_AHEAD = 1.0   # m ahead of robot (look-at target)
CAM_FOV_DEG = 60       # vertical FOV
FOG_DIST = 12.0
NEAR_CLIP = 0.1
FAR_CLIP = 50.0
SSAO_RADIUS = 3.0      # sample radius in pixels
SSAO_INTENSITY = 0.6   # darkening strength
PROX_RADIUS = 0.8      # safety proximity visualisation radius (m)

MINIMAP_SZ = 200
MINIMAP_PAD = 8
MINIMAP_ZOOM = 1.0

PX_SIZE = 0.02
FREE_THRESH = 190
OBS_THRESH = 90

# ── Shaders ──────────────────────────────────────────────────────

_VERT_TERRAIN = """
#version 330

uniform mat4  u_mvp;
uniform sampler2D u_blob;
uniform sampler2D u_conf;
uniform vec2  u_origin;
uniform float u_px;
uniform float u_hscale;
uniform float u_thresh;
uniform ivec2 u_grid;
uniform int   u_gdiv;
uniform int   u_topdown;

in vec2 in_uv;

out vec3 v_col;
out vec3 v_nrm;
out vec3 v_w;
out vec2 v_uv;

void main() {
    float raw_h = texture(u_blob, in_uv).r * 255.0;
    float conf  = texture(u_conf, in_uv).r * 255.0;

    float h_cm = raw_h > u_thresh ? 10.0 : 0.0;
    float h_m  = h_cm * u_hscale;

    float mw = float(u_grid.x * u_gdiv);
    float mh = float(u_grid.y * u_gdiv);
    float wx = (in_uv.x * mw - u_origin.x) * u_px;
    float wz = (in_uv.y * mh - u_origin.y) * u_px;

    vec3 p = vec3(wx, h_m, wz);

    vec2 ts = 1.0 / vec2(u_grid);
    float hL = texture(u_blob, in_uv - vec2(ts.x, 0)).r * 255.0;
    float hR = texture(u_blob, in_uv + vec2(ts.x, 0)).r * 255.0;
    float hD = texture(u_blob, in_uv - vec2(0, ts.y)).r * 255.0;
    float hU = texture(u_blob, in_uv + vec2(0, ts.y)).r * 255.0;
    hL = hL > u_thresh ? 10.0 : 0.0;
    hR = hR > u_thresh ? 10.0 : 0.0;
    hD = hD > u_thresh ? 10.0 : 0.0;
    hU = hU > u_thresh ? 10.0 : 0.0;

    float cell_m = float(u_gdiv) * u_px;
    float dx = (hR - hL) * u_hscale / (2.0 * cell_m);
    float dz = (hU - hD) * u_hscale / (2.0 * cell_m);
    vec3 nrm = normalize(vec3(-dx, 1.0, -dz));

    vec3 col;
    if (h_cm > 0.0) {
        col = u_topdown == 1 ? vec3(0.55, 0.45, 0.3) : vec3(0.55);
    } else if (conf > 190.0) {
        col = u_topdown == 1 ? vec3(1.0) : vec3(0.92);
    } else {
        col = u_topdown == 1 ? vec3(0.4) : vec3(0.25);
    }

    v_col = col;
    v_nrm = nrm;
    v_w   = p;
    v_uv  = in_uv;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

_FRAG_TERRAIN = """
#version 330

uniform sampler2D u_conf;
uniform vec3  u_cam;
uniform float u_fogfar;
uniform int   u_topdown;

in vec3 v_col;
in vec3 v_nrm;
in vec3 v_w;
in vec2 v_uv;

out vec4 fc;

void main() {
    float conf = texture(u_conf, v_uv).r * 255.0;

    if (conf >= 90.0 && conf <= 190.0) {
        vec2 ts = 1.0 / vec2(textureSize(u_conf, 0));
        float cn = texture(u_conf, v_uv + vec2(0, ts.y)).r * 255.0;
        float cs = texture(u_conf, v_uv - vec2(0, ts.y)).r * 255.0;
        float ce = texture(u_conf, v_uv + vec2(ts.x, 0)).r * 255.0;
        float cw = texture(u_conf, v_uv - vec2(ts.x, 0)).r * 255.0;
        bool border = (cn > 190.0 || cn < 90.0) ||
                      (cs > 190.0 || cs < 90.0) ||
                      (ce > 190.0 || ce < 90.0) ||
                      (cw > 190.0 || cw < 90.0);
        if (!border) discard;
    }

    vec3 col = v_col;

    if (u_topdown == 0) {
        vec3 ld = normalize(vec3(0.3, -0.8, -0.5));
        float nl = max(dot(v_nrm, -ld), 0.0);
        col *= (0.35 + 0.65 * nl);

        float d = length(v_w - u_cam);
        float fog = clamp(d / u_fogfar, 0.0, 0.75);
        col = mix(col, vec3(0.0), fog);
    }

    fc = vec4(col, 1.0);
}
"""

_FRAG_BLOB_BLUR = """
#version 330

uniform sampler2D u_in;
uniform sampler2D u_conf;
uniform vec2 u_texel;
uniform int u_mode;
uniform float u_sigma;

out vec4 fc;

void main() {
    vec2 uv = gl_FragCoord.xy * u_texel;
    float v;

    if (u_mode == 0) {
        float c = texture(u_conf, uv).r * 255.0;
        float h = c < 90.0 ? texture(u_in, uv).r * 255.0 : 0.0;
        vec2 uvL = uv - vec2(u_texel.x, 0);
        vec2 uvR = uv + vec2(u_texel.x, 0);
        float cL = texture(u_conf, uvL).r * 255.0;
        float cR = texture(u_conf, uvR).r * 255.0;
        float hL = cL < 90.0 ? texture(u_in, uvL).r * 255.0 : 0.0;
        float hR = cR < 90.0 ? texture(u_in, uvR).r * 255.0 : 0.0;
        v = max(h, max(hL, hR));
    } else if (u_mode == 1) {
        float h  = texture(u_in, uv).r * 255.0;
        float hU = texture(u_in, uv - vec2(0, u_texel.y)).r * 255.0;
        float hD = texture(u_in, uv + vec2(0, u_texel.y)).r * 255.0;
        v = max(h, max(hU, hD));
    } else {
        vec2 dir = u_mode == 2 ? vec2(u_texel.x, 0) : vec2(0, u_texel.y);
        float sum = 0.0, wt = 0.0;
        for (int i = -2; i <= 2; i++) {
            float g = exp(-0.5 * float(i*i) / (u_sigma * u_sigma));
            float s = texture(u_in, uv + float(i) * dir).r * 255.0;
            sum += s * g;
            wt += g;
        }
        v = sum / wt;
    }

    fc = vec4(v / 255.0);
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

# ── Shaders for atlas overlays (cameras, minimap, battery, trail) ─

_FRAG_CAMERA = """
#version 330
uniform sampler2D u_tex;
uniform vec4 u_vp;
out vec4 fc;
void main() {
    vec2 local = (gl_FragCoord.xy - u_vp.xy) / u_vp.zw;
    fc = texture(u_tex, vec2(local.x, 1.0 - local.y));
}
"""

_FRAG_MINIMAP = """
#version 330
uniform sampler2D u_conf;
uniform sampler2D u_hmap;
uniform vec4 u_vp;
uniform vec2 u_center;
uniform float u_zoom;
uniform vec2 u_mapsz;
uniform float u_sf;
uniform float u_theta;
out vec4 fc;
void main() {
    vec2 local = (gl_FragCoord.xy - u_vp.xy) / u_vp.zw;
    /* Border — 2px, fully opaque, drawn first */
    float bw = 2.0 / u_vp.z;
    if (local.x < bw || local.x > 1.0 - bw ||
        local.y < bw || local.y > 1.0 - bw) {
        fc = vec4(0.314, 0.314, 0.314, 1.0);
        return;
    }
    /* Swap axes so +x (east/initial heading) points up on screen */
    vec2 off = local - 0.5;
    vec2 rot_off = vec2(off.y, off.x);
    vec2 map_uv = rot_off / u_zoom + u_center / u_mapsz;
    if (map_uv.x < 0.0 || map_uv.x > 1.0 ||
        map_uv.y < 0.0 || map_uv.y > 1.0) {
        fc = vec4(0.627, 0.627, 0.627, 0.85);
        return;
    }
    float c = texture(u_conf, map_uv).r * 255.0;
    vec3 col;
    if (c > 190.0) {
        col = vec3(1.0);
    } else if (c < 90.0) {
        float h = texture(u_hmap, map_uv).r * 255.0;
        float g = clamp((64.0 - h * 0.5) / 255.0, 0.04, 0.25);
        col = vec3(g);
    } else {
        col = vec3(0.627);
    }
    /* Robot dot at center */
    vec2 px = local * u_vp.zw;
    vec2 ctr = u_vp.zw * 0.5;
    float d = length(px - ctr);
    if (d < 4.0) {
        float r_col = mix(1.0, 0.0, u_sf);
        float g_col = mix(0.0, 0.78, u_sf);
        float b_col = mix(0.0, 1.0, u_sf);
        col = vec3(r_col, g_col, b_col);
    }
    /* Direction tick — matches rotated minimap */
    vec2 dir = vec2(-sin(u_theta), cos(u_theta));  /* map fwd (cosθ,-sinθ) → screen via inverse swap */
    vec2 dd = px - ctr;
    float along = dot(dd, dir);
    float perp = abs(dd.x * dir.y - dd.y * dir.x);
    if (along > 3.0 && along < 8.0 && perp < 1.5) {
        float r_col = mix(1.0, 0.0, u_sf);
        float g_col = mix(0.0, 0.78, u_sf);
        col = vec3(r_col, g_col, 1.0);
    }
    fc = vec4(col, 0.85);
}
"""

_FRAG_BATTERY = """
#version 330
uniform vec4 u_vp;
uniform float u_frac;
out vec4 fc;
void main() {
    vec2 local = (gl_FragCoord.xy - u_vp.xy) / u_vp.zw;
    float fill = step(local.x, u_frac);
    vec3 bg = vec3(0.118);
    vec3 col;
    if (u_frac > 0.50) col = vec3(0.0, 0.78, 0.0);
    else if (u_frac > 0.25) col = vec3(0.86, 0.7, 0.0);
    else col = vec3(0.86, 0.16, 0.16);
    fc = vec4(mix(bg, col, fill), 1.0);
}
"""

_VERT_TRAIL = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

_FRAG_TRAIL = """
#version 330
out vec4 fc;
void main() {
    fc = vec4(0.314, 0.549, 1.0, 1.0);
}
"""

# ── SSAO post-process shaders ────────────────────────────────────

_VERT_SSAO = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

_FRAG_SSAO = """
#version 330

uniform sampler2D u_scene;
uniform sampler2D u_depth;
uniform vec2  u_texel;
uniform float u_near;
uniform float u_far;
uniform float u_radius;
uniform float u_intensity;

uniform mat4  u_inv_vp;
uniform vec3  u_robot_pos;
uniform float u_heading;
uniform float u_fwd_scale;
uniform float u_bwd_scale;
uniform float u_ang_scale;
uniform float u_prox_radius;

in vec2 v_uv;
out vec4 fc;

float linearize(float d) {
    return u_near * u_far / (u_far - d * (u_far - u_near));
}

const int N = 8;
const vec2 kernel[8] = vec2[](
    vec2(-0.7071, -0.7071), vec2( 0.7071, -0.7071),
    vec2(-0.7071,  0.7071), vec2( 0.7071,  0.7071),
    vec2(-1.0,  0.0),       vec2( 1.0,  0.0),
    vec2( 0.0, -1.0),       vec2( 0.0,  1.0)
);

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    float d = texture(u_depth, v_uv).r;

    if (d >= 0.999) {
        fc = vec4(scene, 1.0);
        return;
    }

    /* ── SSAO ── */
    float ao = 1.0;
    if (u_intensity > 0.0) {
        float depth = linearize(d);
        float occ = 0.0;
        for (int i = 0; i < N; i++) {
            vec2 off = kernel[i] * u_radius * u_texel;
            float sd = linearize(texture(u_depth, v_uv + off).r);
            float diff = depth - sd;
            if (diff > 0.005)
                occ += smoothstep(0.0, 0.3, diff);
        }
        ao = clamp(1.0 - (occ / float(N)) * u_intensity, 0.25, 1.0);
    }
    vec3 col = scene * ao;

    /* ── Proximity / throttle visualisation ── */
    if (u_prox_radius > 0.0) {
        vec4 ndc = vec4(v_uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
        vec4 wp = u_inv_vp * ndc;
        wp /= wp.w;

        vec2 delta = wp.xz - u_robot_pos.xz;
        float dist = length(delta);

        if (dist < u_prox_radius && dist > 0.06) {
            float ct = cos(u_heading);
            float st = sin(u_heading);
            float fwd = delta.x * ct - delta.y * st;
            float lat = -delta.x * st - delta.y * ct;

            float scale;
            if (abs(fwd) > abs(lat))
                scale = fwd > 0.0 ? u_fwd_scale : u_bwd_scale;
            else
                scale = u_ang_scale;

            float throttle = 1.0 - scale;
            if (throttle > 0.05) {
                float prox = 1.0 - dist / u_prox_radius;
                float alpha = throttle * prox * 0.55;
                vec3 warn = mix(vec3(1.0, 1.0, 0.0),
                                vec3(1.0, 0.0, 0.0), throttle);
                col = mix(col, warn, alpha);
            }
        }
    }

    fc = vec4(col, 1.0);
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
    v_h = 0.0;
    gl_PointSize = 1.0;
    vec3 p = in_v;
    if (p.z <= 0.0) { gl_Position = vec4(2.0,2.0,0.0,1.0); return; }
    p.y += u_y_off;
    vec3 r = u_rot * (p - u_pivot) + u_pivot - u_trans;
    float enc;
    if (r.z > u_floor && r.z < u_ceil) {
        float phys_h = u_cam_h - p.y * u_cos_p - p.z * u_sin_p;
        if (phys_h > 0.03) {
            enc = clamp(phys_h * 100.0, 3.0, 100.0) + 1.0;
        } else {
            enc = 1.0;
            gl_PointSize = 3.0;
        }
    } else {
        enc = 1.0;
        gl_PointSize = 3.0;
    }
    vec2 px = r.xy * u_scale + u_offset;
    vec2 ndc = px / u_fbo_sz * 2.0 - 1.0;
    gl_Position = vec4(ndc, enc / 52.0 - 1.0, 1.0);
    v_h = enc;
}
"""

_FRAG_SCATTER_OBS = """
#version 330
flat in float v_h;
out vec4 fc;
void main() { fc = vec4(v_h / 255.0, 0.0, 0.0, 1.0); }
"""

_VERT_FSQUAD = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# ── Morph close shaders ──────────────────────────────────────────

_FRAG_MORPH = """
#version 330
uniform sampler2D u_in;
uniform vec2 u_texel;
uniform int u_mode;
out vec4 fc;
void main() {
    vec2 uv = gl_FragCoord.xy * u_texel;
    float v;
    if (u_mode == 0) {
        v = 0.0;
        v=max(v,texture(u_in,uv+vec2( 0,-1)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 1,-1)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2(-1, 0)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 0, 0)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 1, 0)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 2, 0)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2(-1, 1)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 0, 1)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 1, 1)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 2, 1)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 0, 2)*u_texel).r);
        v=max(v,texture(u_in,uv+vec2( 1, 2)*u_texel).r);
        fc = vec4(v > 0.006 ? 1.0 : 0.0);
    } else {
        v = 1.0;
        v=min(v,texture(u_in,uv+vec2( 0,-1)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 1,-1)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2(-1, 0)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 0, 0)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 1, 0)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 2, 0)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2(-1, 1)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 0, 1)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 1, 1)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 2, 1)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 0, 2)*u_texel).r);
        v=min(v,texture(u_in,uv+vec2( 1, 2)*u_texel).r);
        fc = vec4(v > 0.5 ? 1.0 : 0.0);
    }
}
"""

_FRAG_OBS_COMBINE = """
#version 330
uniform sampler2D u_heights;
uniform sampler2D u_morph;
uniform vec2 u_texel;
out vec4 fc;
void main() {
    vec2 uv = gl_FragCoord.xy * u_texel;
    float h = texture(u_heights, uv).r;
    float m = texture(u_morph, uv).r;
    fc = vec4(h > 0.006 ? h : (m > 0.5 ? 0.00784 : h));
}
"""

# ── GPU visual odometry shaders ──────────────────────────────────

_FRAG_DOWNSAMPLE = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_inv_dst;
out vec4 fc;
void main() {
    fc = vec4(texture(u_src, gl_FragCoord.xy * u_inv_dst).r);
}
"""

_FRAG_SAD_SEARCH = """
#version 330
uniform sampler2D u_prev;
uniform sampler2D u_curr;
uniform float u_search;
uniform vec2 u_ds_inv;
uniform ivec2 u_ds_size;
out vec4 fc;
void main() {
    vec2 idx = floor(gl_FragCoord.xy - 0.5);
    vec2 offset = idx - vec2(u_search);
    float sad = 0.0;
    float count = 0.0;
    for (int y = 0; y < u_ds_size.y; y++) {
        for (int x = 0; x < u_ds_size.x; x++) {
            vec2 puv = (vec2(x, y) + 0.5) * u_ds_inv;
            vec2 cpx = vec2(x, y) + offset;
            if (cpx.x >= 0.0 && cpx.x < float(u_ds_size.x) &&
                cpx.y >= 0.0 && cpx.y < float(u_ds_size.y)) {
                sad += abs(texture(u_prev, puv).r -
                           texture(u_curr, (cpx+0.5)*u_ds_inv).r);
                count += 1.0;
            }
        }
    }
    fc = vec4(count > 0.0 ? sad / count : 1.0);
}
"""

_FRAG_TEXCOPY = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_inv;
out vec4 fc;
void main() { fc = vec4(texture(u_src, gl_FragCoord.xy * u_inv).r); }
"""

# ── Global-map evidence-update shader (MRT: conf + hmap) ────────

_FRAG_GMAP_UPDATE = """
#version 330
uniform sampler2D u_conf;
uniform sampler2D u_hmap;
uniform sampler2D u_ego;
uniform mat3 u_inv;
uniform vec2 u_map_inv;
uniform vec2 u_ego_inv;
uniform float u_step_free;
uniform float u_step_obs;

layout(location = 0) out vec4 out_conf;
layout(location = 1) out vec4 out_hmap;

void main() {
    vec2 gp = gl_FragCoord.xy;
    vec2 guv = gp * u_map_inv;
    float conf = texture(u_conf, guv).r;
    float h    = texture(u_hmap, guv).r;

    vec3 ep = u_inv * vec3(gp, 1.0);
    vec2 euv = ep.xy * u_ego_inv;

    if (euv.x > 0.0 && euv.x < 1.0 && euv.y > 0.0 && euv.y < 1.0) {
        float obs_i = floor(texture(u_ego, euv).r * 255.0 + 0.5);
        if (obs_i == 1.0) {
            conf = min(conf + u_step_free, 1.0);
            h = 0.0;
        } else if (obs_i >= 2.0) {
            conf = max(conf - u_step_obs, 0.0);
            float new_h = (obs_i - 1.0) / 255.0;
            h = max(h, new_h);
        }
    }
    out_conf = vec4(conf);
    out_hmap = vec4(h);
}
"""

_FRAG_GMAP_PROJECT = """
#version 330
uniform sampler2D u_conf;
uniform mat3 u_fwd;
uniform vec2 u_map_inv;
uniform float u_unknown;
out vec4 fc;
void main() {
    vec3 gp = u_fwd * vec3(gl_FragCoord.xy, 1.0);
    vec2 guv = gp.xy * u_map_inv;
    if (guv.x >= 0.0 && guv.x <= 1.0 && guv.y >= 0.0 && guv.y <= 1.0)
        fc = vec4(texture(u_conf, guv).r);
    else
        fc = vec4(u_unknown);
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


# ── Voxel cube geometry ──────────────────────────────────────────

def _make_cube_data():
    """Unit cube [-0.5, 0.5]^3 with per-face normals (24 verts, 36 indices)."""
    P = 0.5
    faces = [
        ([(-P, P,-P), ( P, P,-P), ( P, P, P), (-P, P, P)], ( 0, 1, 0)),  # top
        ([(-P,-P, P), ( P,-P, P), ( P,-P,-P), (-P,-P,-P)], ( 0,-1, 0)),  # bottom
        ([(-P,-P, P), (-P, P, P), ( P, P, P), ( P,-P, P)], ( 0, 0, 1)),  # front
        ([( P,-P,-P), ( P, P,-P), (-P, P,-P), (-P,-P,-P)], ( 0, 0,-1)),  # back
        ([( P,-P, P), ( P, P, P), ( P, P,-P), ( P,-P,-P)], ( 1, 0, 0)),  # right
        ([(-P,-P,-P), (-P, P,-P), (-P, P, P), (-P,-P, P)], (-1, 0, 0)),  # left
    ]
    verts = []
    idx = []
    for fi, (fv, n) in enumerate(faces):
        base = fi * 4
        for v in fv:
            verts.append((*v, *n))
        idx.extend([base, base+1, base+2, base, base+2, base+3])
    return (np.array(verts, dtype=np.float32),
            np.array(idx, dtype=np.int32))


def _make_grid_data(gw, gh):
    """UV grid mesh for displacement terrain. Returns (float32 verts, int32 indices)."""
    nvx, nvy = gw + 1, gh + 1
    u = np.linspace(0, 1, nvx, dtype=np.float32)
    v = np.linspace(0, 1, nvy, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    verts = np.stack([uu.ravel(), vv.ravel()], axis=-1)

    iy, ix = np.meshgrid(
        np.arange(gh, dtype=np.int32),
        np.arange(gw, dtype=np.int32), indexing='ij')
    i0 = (iy * nvx + ix).ravel()
    i1 = i0 + 1
    i2 = i0 + nvx
    i3 = i2 + 1
    indices = np.column_stack([i0, i2, i1, i1, i2, i3]).ravel()
    return verts, indices


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
    """GPU-accelerated full-atlas renderer (3D voxel terrain + SSAO + HUD).

    atlas_w × atlas_h = full output size (e.g. 960×960).
    map_w × map_h = occupancy grid size (e.g. 960×720).
    The 3D voxel terrain occupies the lower map_h rows; cameras fill the top.
    """

    CAM_H = 240
    CAM_W = 320
    BAR_H = 6

    def __init__(self, map_w, map_h, atlas_w, atlas_h):
        if not _HAS_MGL:
            raise RuntimeError("moderngl not installed — GPU required")
        self.available = True
        self._mw = map_w
        self._mh = map_h
        self._vw = atlas_w
        self._vh = atlas_h
        self._view3d_h = map_h
        self.topdown = False
        self._gl_ready = False

    # ── Depth-forward configuration (call once before use) ───────

    def configure_depth_forward(self, rotation, pivot, translation,
                                 px_size, cam_height, sin_pitch, cos_pitch,
                                 floor_clip, height_clip,
                                 out_h, out_w):
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
        self._df_configured = True
        self._df_gl_ready = False

    # ── GL init ──────────────────────────────────────────────────

    def _init_gl(self):
        t0 = time.monotonic()
        ctx = self._create_context()

        # Main FBO at full atlas size
        self._color_tex_fbo = ctx.texture((self._vw, self._vh), 4)
        depth = ctx.depth_renderbuffer((self._vw, self._vh))
        self._fbo = ctx.framebuffer(
            color_attachments=[self._color_tex_fbo],
            depth_attachment=depth)

        # Scene FBO for SSAO pipeline (3D viewport only)
        sw, sh = self._vw, self._view3d_h
        self._scene_color_tex = ctx.texture((sw, sh), 4)
        self._scene_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scene_depth_tex = ctx.depth_texture((sw, sh))
        self._scene_depth_tex.compare_func = ''
        self._scene_fbo = ctx.framebuffer(
            color_attachments=[self._scene_color_tex],
            depth_attachment=self._scene_depth_tex)

        # Fullscreen-quad VBO (shared by all overlay / blur shaders)
        fsq = np.float32([-1, -1, 1, -1, -1, 1, 1, 1])
        self._fsq_vbo = ctx.buffer(fsq.tobytes())

        # Blob textures for smooth heightmap (at BLOB_DIV resolution)
        bw = self._mw // BLOB_DIV
        bh = self._mh // BLOB_DIV
        self._blob_a_tex = ctx.texture((bw, bh), 1)
        self._blob_a_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._blob_a_fbo = ctx.framebuffer(
            color_attachments=[self._blob_a_tex])
        self._blob_b_tex = ctx.texture((bw, bh), 1)
        self._blob_b_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._blob_b_fbo = ctx.framebuffer(
            color_attachments=[self._blob_b_tex])

        self._prog_bblur = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_BLOB_BLUR)
        self._prog_bblur['u_in'].value = 0
        self._prog_bblur['u_conf'].value = 1
        self._prog_bblur['u_texel'].value = (1.0 / bw, 1.0 / bh)
        self._prog_bblur['u_sigma'].value = BLOB_SIGMA
        self._vao_bblur = ctx.vertex_array(
            self._prog_bblur, [(self._fsq_vbo, '2f', 'in_pos')])

        # Displacement terrain shader (smooth surface over blob heightmap)
        gw = self._mw // GRID_DIV
        gh = self._mh // GRID_DIV

        self._prog_t = ctx.program(
            vertex_shader=_VERT_TERRAIN, fragment_shader=_FRAG_TERRAIN)
        self._prog_t['u_origin'].value = (
            float(self._mw // 2), float(self._mh // 2))
        self._prog_t['u_px'].value = PX_SIZE
        self._prog_t['u_hscale'].value = 0.01
        self._prog_t['u_thresh'].value = BLOB_THRESH
        self._prog_t['u_fogfar'].value = FOG_DIST
        self._prog_t['u_blob'].value = 0
        self._prog_t['u_conf'].value = 1
        self._prog_t['u_topdown'].value = 0
        self._prog_t['u_grid'].value = (gw, gh)
        self._prog_t['u_gdiv'].value = GRID_DIV
        self._build_terrain()

        # Robot shader + mesh
        self._prog_r = ctx.program(
            vertex_shader=_VERT_ROBOT, fragment_shader=_FRAG_ROBOT)
        self._build_robot()

        # Camera feed textures (3 cameras, uploaded each frame)
        self._cam_textures = [
            ctx.texture((self.CAM_W, self.CAM_H), 3) for _ in range(3)]
        for t in self._cam_textures:
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._prog_cam = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_CAMERA)
        self._vao_cam = ctx.vertex_array(
            self._prog_cam, [(self._fsq_vbo, '2f', 'in_pos')])

        # Minimap shader
        self._prog_mm = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_MINIMAP)
        self._vao_mm = ctx.vertex_array(
            self._prog_mm, [(self._fsq_vbo, '2f', 'in_pos')])

        # Battery bar shader
        self._prog_bat = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_BATTERY)
        self._vao_bat = ctx.vertex_array(
            self._prog_bat, [(self._fsq_vbo, '2f', 'in_pos')])

        # Trail line shader
        self._prog_trail = ctx.program(
            vertex_shader=_VERT_TRAIL, fragment_shader=_FRAG_TRAIL)
        self._trail_vbo = ctx.buffer(reserve=300 * 3 * 4)
        self._vao_trail = ctx.vertex_array(
            self._prog_trail, [(self._trail_vbo, '3f', 'in_pos')])

        # SSAO + proximity post-process shader
        self._prog_ssao = ctx.program(
            vertex_shader=_VERT_SSAO, fragment_shader=_FRAG_SSAO)
        self._prog_ssao['u_scene'].value = 0
        self._prog_ssao['u_depth'].value = 1
        self._prog_ssao['u_texel'].value = (1.0 / sw, 1.0 / sh)
        self._prog_ssao['u_near'].value = NEAR_CLIP
        self._prog_ssao['u_far'].value = FAR_CLIP
        self._prog_ssao['u_radius'].value = SSAO_RADIUS
        self._prog_ssao['u_intensity'].value = SSAO_INTENSITY
        self._prog_ssao['u_prox_radius'].value = PROX_RADIUS
        self._vao_ssao = ctx.vertex_array(
            self._prog_ssao, [(self._fsq_vbo, '2f', 'in_pos')])

        # Pre-allocated readback buffer
        self._out = np.empty((self._vh, self._vw, 3), dtype=np.uint8)

        # PBO double-buffer for async readback
        pbo_sz = self._vw * self._vh * 3
        self._pbo = [ctx.buffer(reserve=pbo_sz),
                     ctx.buffer(reserve=pbo_sz)]
        self._pbo_idx = 0
        self._pbo_ready = False

        # Initialize gmap GL if already configured
        if getattr(self, '_gm_configured', False) and not getattr(self, '_gm_gl_ready', False):
            self._init_gmap_gl()

        # Initialize depth GL if already configured
        if getattr(self, '_df_configured', False) and not getattr(self, '_df_gl_ready', False):
            self._init_depth_gl()

        # Initialize odom GL if already configured
        if getattr(self, '_od_configured', False) and not getattr(self, '_od_gl_ready', False):
            self._init_odom_gl()

        t1 = time.monotonic()
        bw = self._mw // BLOB_DIV
        bh = self._mh // BLOB_DIV
        print("gpu_render: ready %dx%d  mesh=%dx%d  blob=%dx%d  robot=%d tris  %.0fms"
              % (self._vw, self._vh, self._gw, self._gh, bw, bh,
                 self._robot_ntris, (t1 - t0) * 1e3))

    def _create_context(self):
        try:
            self._ctx = moderngl.create_context(
                standalone=True, backend='egl')
        except Exception:
            self._ctx = moderngl.create_context(standalone=True)
        return self._ctx

    def _build_terrain(self):
        gw = self._mw // GRID_DIV
        gh = self._mh // GRID_DIV
        self._gw, self._gh = gw, gh

        verts, idx = _make_grid_data(gw, gh)
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

        self._df_obs_tex = ctx.texture((ow, oh), 1)
        self._df_obs_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._df_obs_depth = ctx.depth_renderbuffer((ow, oh))
        self._df_obs_fbo = ctx.framebuffer(
            color_attachments=[self._df_obs_tex],
            depth_attachment=self._df_obs_depth)

        self._df_morph_a_tex = ctx.texture((ow, oh), 1)
        self._df_morph_a_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._df_morph_a_fbo = ctx.framebuffer(
            color_attachments=[self._df_morph_a_tex])

        self._df_morph_b_tex = ctx.texture((ow, oh), 1)
        self._df_morph_b_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._df_morph_b_fbo = ctx.framebuffer(
            color_attachments=[self._df_morph_b_tex])

        max_pts = 320 * 240
        self._df_vbo = ctx.buffer(reserve=max_pts * 12)

        self._df_prog_obs = ctx.program(
            vertex_shader=_VERT_SCATTER_OBS,
            fragment_shader=_FRAG_SCATTER_OBS)
        self._df_prog_morph = ctx.program(
            vertex_shader=_VERT_FSQUAD,
            fragment_shader=_FRAG_MORPH)
        self._df_prog_combine = ctx.program(
            vertex_shader=_VERT_FSQUAD,
            fragment_shader=_FRAG_OBS_COMBINE)

        self._df_vao_obs = ctx.vertex_array(
            self._df_prog_obs, [(self._df_vbo, '3f', 'in_v')])

        fsq_buf = self._fsq_vbo
        self._df_vao_morph = ctx.vertex_array(
            self._df_prog_morph, [(fsq_buf, '2f', 'in_pos')])
        self._df_vao_combine = ctx.vertex_array(
            self._df_prog_combine, [(fsq_buf, '2f', 'in_pos')])

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

        df_texel = (1.0 / float(ow), 1.0 / float(oh))
        p = self._df_prog_morph
        p['u_in'].value = 0
        p['u_texel'].value = df_texel
        p = self._df_prog_combine
        p['u_heights'].value = 0
        p['u_morph'].value = 1
        p['u_texel'].value = df_texel

        try:
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        except Exception:
            pass

        self._df_gl_ready = True
        self._df_n = 0
        print("gpu_render: depth_forward ready %dx%d (floor-as-free)"
              % (ow, oh))

    # ── GPU depth-forward pipeline ────────────────────────────────

    def depth_forward_gpu(self, verts, y_offset=0.0, debug=False):
        """Process RS2 forward depth on GPU: scatter (floor=free) + morph.
        Returns (obs, known, raw_scatter) — raw_scatter is None unless debug=True.
        """
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

        # Scatter all valid points → obs FBO (max-encode via depth test)
        self._df_obs_fbo.use()
        self._df_obs_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=0.0)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_func = '>'
        self._df_vao_obs.render(moderngl.POINTS, vertices=n_pts)

        raw_scatter = None
        if debug:
            raw_scat_data = self._df_obs_fbo.read(components=1, alignment=1)
            raw_scatter = np.frombuffer(raw_scat_data, dtype=np.uint8).reshape(oh, ow)

        # GPU morph close on obstacles only: dilate×2, erode×2 (ping-pong)
        self._ctx.disable(moderngl.DEPTH_TEST)
        morph_src = [self._df_obs_tex, self._df_morph_a_tex,
                     self._df_morph_b_tex, self._df_morph_a_tex]
        morph_dst = [self._df_morph_a_fbo, self._df_morph_b_fbo,
                     self._df_morph_a_fbo, self._df_morph_b_fbo]
        morph_mode = [0, 0, 1, 1]
        for src_tex, dst_fbo, mode in zip(morph_src, morph_dst, morph_mode):
            dst_fbo.use()
            dst_fbo.clear(0.0, 0.0, 0.0, 0.0)
            src_tex.use(location=0)
            self._df_prog_morph['u_mode'].value = mode
            self._df_vao_morph.render(moderngl.TRIANGLE_STRIP)
        t1 = time.monotonic()

        # Combine: original encoded values + morph binary → morph_a
        self._df_morph_a_fbo.use()
        self._df_morph_a_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._df_obs_tex.use(location=0)
        self._df_morph_b_tex.use(location=1)
        self._df_vao_combine.render(moderngl.TRIANGLE_STRIP)

        # Single readback — decode floor-as-free encoding to (obs, known)
        raw_data = self._df_morph_a_fbo.read(components=1, alignment=1)
        raw = np.frombuffer(raw_data, dtype=np.uint8).reshape(oh, ow)
        known = np.where(raw > 0, np.uint8(255), np.uint8(0))
        obs = np.where(raw >= 2, (raw - 1).astype(np.uint8), np.uint8(0))

        self._ctx.depth_func = '<'

        self._df_n += 1
        t2 = time.monotonic()
        if self._df_n <= 3 or self._df_n % 100 == 0:
            n_empty = int(np.count_nonzero(raw == 0))
            n_floor = int(np.count_nonzero(raw == 1))
            n_obs = int(np.count_nonzero(raw >= 2))
            print("gpu_depth: scatter+morph=%.1fms read+decode=%.1fms "
                  "total=%.1fms  floor=%d obs=%d empty=%d" % (
                      (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3,
                      n_floor, n_obs, n_empty))

        return obs, known, raw_scatter

    # ── GPU visual odometry ─────────────────────────────────────

    def configure_odom(self, fx=307.0, ds_factor=4, search=8):
        self._od_fx = float(fx)
        self._od_ds = int(ds_factor)
        self._od_search = int(search)
        self._od_configured = True
        self._od_gl_ready = False

    def _init_odom_gl(self):
        ctx = self._ctx
        ds = self._od_ds
        sw = 2 * self._od_search + 1
        dsw, dsh = 320 // ds, 240 // ds

        self._od_gray_tex = ctx.texture((320, 240), 1)
        self._od_gray_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._od_ds_prev_tex = ctx.texture((dsw, dsh), 1)
        self._od_ds_prev_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._od_ds_prev_fbo = ctx.framebuffer(
            color_attachments=[self._od_ds_prev_tex])

        self._od_ds_curr_tex = ctx.texture((dsw, dsh), 1)
        self._od_ds_curr_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._od_ds_curr_fbo = ctx.framebuffer(
            color_attachments=[self._od_ds_curr_tex])

        self._od_sad_tex = ctx.texture((sw, sw), 1, dtype='f4')
        self._od_sad_fbo = ctx.framebuffer(
            color_attachments=[self._od_sad_tex])

        self._od_prog_ds = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_DOWNSAMPLE)
        self._od_prog_sad = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_SAD_SEARCH)
        self._od_prog_copy = ctx.program(
            vertex_shader=_VERT_FSQUAD, fragment_shader=_FRAG_TEXCOPY)

        fsq_buf = self._fsq_vbo
        self._od_vao_ds = ctx.vertex_array(
            self._od_prog_ds, [(fsq_buf, '2f', 'in_pos')])
        self._od_vao_sad = ctx.vertex_array(
            self._od_prog_sad, [(fsq_buf, '2f', 'in_pos')])
        self._od_vao_copy = ctx.vertex_array(
            self._od_prog_copy, [(fsq_buf, '2f', 'in_pos')])

        self._od_prog_ds['u_src'].value = 0
        self._od_prog_ds['u_inv_dst'].value = (1.0 / dsw, 1.0 / dsh)

        self._od_prog_sad['u_prev'].value = 0
        self._od_prog_sad['u_curr'].value = 1
        self._od_prog_sad['u_search'].value = float(self._od_search)
        self._od_prog_sad['u_ds_inv'].value = (1.0 / dsw, 1.0 / dsh)
        self._od_prog_sad['u_ds_size'].value = (dsw, dsh)

        self._od_prog_copy['u_src'].value = 0
        self._od_prog_copy['u_inv'].value = (1.0 / dsw, 1.0 / dsh)

        self._od_dsw, self._od_dsh = dsw, dsh
        self._od_sw = sw
        self._od_has_prev = False
        self._od_gl_ready = True
        self._od_n = 0
        print("gpu_render: odom ready ds=%dx%d  search=±%d" % (
            dsw, dsh, self._od_search))

    def odom_gpu(self, gray):
        """GPU visual odometry: SAD-based yaw estimation.
        Returns (yaw_rad, forward_m, confidence) or None."""
        if not self.available or not getattr(self, '_od_configured', False):
            return None
        if not self._gl_ready:
            return None
        if not getattr(self, '_od_gl_ready', False):
            try:
                self._init_odom_gl()
            except Exception as e:
                print("gpu_render: odom init failed: %s" % e)
                import traceback; traceback.print_exc()
                return None

        t0 = time.monotonic()

        self._od_gray_tex.write(gray.tobytes())

        self._od_ds_curr_fbo.use()
        self._od_gray_tex.use(location=0)
        self._od_vao_ds.render(moderngl.TRIANGLE_STRIP)

        if not self._od_has_prev:
            self._od_ds_prev_fbo.use()
            self._od_ds_curr_tex.use(location=0)
            self._od_vao_copy.render(moderngl.TRIANGLE_STRIP)
            self._od_has_prev = True
            return 0.0, 0.0, 0.0

        self._od_sad_fbo.use()
        self._od_ds_prev_tex.use(location=0)
        self._od_ds_curr_tex.use(location=1)
        self._od_vao_sad.render(moderngl.TRIANGLE_STRIP)

        sad_data = self._od_sad_fbo.read(
            components=1, alignment=4, dtype='f4')
        sw = self._od_sw
        sad = np.frombuffer(sad_data, dtype=np.float32).reshape(sw, sw).copy()

        # Copy curr → prev for next frame
        self._od_ds_prev_fbo.use()
        self._od_ds_curr_tex.use(location=0)
        self._od_vao_copy.render(moderngl.TRIANGLE_STRIP)

        t1 = time.monotonic()

        # Find minimum + sub-pixel refinement
        sr = self._od_search
        min_idx = np.unravel_index(np.argmin(sad), sad.shape)
        y0, x0 = int(min_idx[0]), int(min_idx[1])
        sub_x = float(x0 - sr)
        sub_y = float(y0 - sr)
        if 0 < x0 < sw - 1:
            denom = 2.0 * sad[y0, x0] - sad[y0, x0 - 1] - sad[y0, x0 + 1]
            if abs(denom) > 1e-8:
                sub_x += (sad[y0, x0 - 1] - sad[y0, x0 + 1]) / (2.0 * denom)
        if 0 < y0 < sw - 1:
            denom = 2.0 * sad[y0, x0] - sad[y0 - 1, x0] - sad[y0 + 1, x0]
            if abs(denom) > 1e-8:
                sub_y += (sad[y0 - 1, x0] - sad[y0 + 1, x0]) / (2.0 * denom)

        yaw = sub_x * self._od_ds / self._od_fx
        forward = 0.0

        min_sad = float(sad.min())
        mean_sad = float(sad.mean())
        sharpness = (mean_sad - min_sad) / (mean_sad + 1e-6)
        confidence = min(1.0, sharpness * 2.0)

        self._od_n += 1
        if self._od_n <= 3 or self._od_n % 100 == 0:
            print("gpu_odom: %.1fms  dx=%.2f dy=%.2f yaw=%.3f° conf=%.2f" % (
                (t1 - t0) * 1e3, sub_x, sub_y,
                math.degrees(yaw), confidence))

        return yaw, forward, confidence

    # ── GPU global-map evidence update ───────────────────────────

    def configure_gmap(self, map_w, map_h, ego_w, ego_h,
                       origin_x, origin_y, px_size,
                       step_free=60, step_obs=25):
        self._gm_mw = int(map_w)
        self._gm_mh = int(map_h)
        self._gm_ew = int(ego_w)
        self._gm_eh = int(ego_h)
        self._gm_ox = float(origin_x)
        self._gm_oy = float(origin_y)
        self._gm_px = float(px_size)
        self._gm_step_free = float(step_free) / 255.0
        self._gm_step_obs = float(step_obs) / 255.0
        self._gm_configured = True
        self._gm_gl_ready = False

    def _init_gmap_gl(self):
        ctx = self._ctx
        mw, mh = self._gm_mw, self._gm_mh
        ew, eh = self._gm_ew, self._gm_eh

        init_conf = np.full(mw * mh, 128, dtype=np.uint8)
        init_hmap = np.zeros(mw * mh, dtype=np.uint8)

        self._gm_conf = [ctx.texture((mw, mh), 1),
                         ctx.texture((mw, mh), 1)]
        self._gm_hmap = [ctx.texture((mw, mh), 1),
                         ctx.texture((mw, mh), 1)]
        for t in self._gm_conf:
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.write(init_conf.tobytes())
        for t in self._gm_hmap:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.write(init_hmap.tobytes())

        self._gm_fbo = [
            ctx.framebuffer(color_attachments=[self._gm_conf[0],
                                               self._gm_hmap[0]]),
            ctx.framebuffer(color_attachments=[self._gm_conf[1],
                                               self._gm_hmap[1]]),
        ]
        self._gm_idx = 0

        self._gm_ego_tex = ctx.texture((ew, eh), 1)
        self._gm_ego_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        fsq = self._fsq_vbo

        self._gm_prog_up = ctx.program(
            vertex_shader=_VERT_FSQUAD,
            fragment_shader=_FRAG_GMAP_UPDATE)
        self._gm_prog_up['u_conf'].value = 0
        self._gm_prog_up['u_hmap'].value = 1
        self._gm_prog_up['u_ego'].value = 2
        self._gm_prog_up['u_map_inv'].value = (1.0 / mw, 1.0 / mh)
        self._gm_prog_up['u_ego_inv'].value = (1.0 / ew, 1.0 / eh)
        self._gm_prog_up['u_step_free'].value = self._gm_step_free
        self._gm_prog_up['u_step_obs'].value = self._gm_step_obs
        self._gm_vao_up = ctx.vertex_array(
            self._gm_prog_up, [(fsq, '2f', 'in_pos')])

        self._gm_proj_tex = ctx.texture((ew, eh), 1)
        self._gm_proj_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._gm_proj_fbo = ctx.framebuffer(
            color_attachments=[self._gm_proj_tex])

        self._gm_prog_pj = ctx.program(
            vertex_shader=_VERT_FSQUAD,
            fragment_shader=_FRAG_GMAP_PROJECT)
        self._gm_prog_pj['u_conf'].value = 0
        self._gm_prog_pj['u_map_inv'].value = (1.0 / mw, 1.0 / mh)
        self._gm_prog_pj['u_unknown'].value = 128.0 / 255.0
        self._gm_vao_pj = ctx.vertex_array(
            self._gm_prog_pj, [(fsq, '2f', 'in_pos')])

        self._gm_n = 0
        self._gm_gl_ready = True
        print("gpu_render: gmap ready %dx%d  step_free=%d step_obs=%d" % (
            mw, mh, int(self._gm_step_free * 255),
            int(self._gm_step_obs * 255)))

    def _gmap_affine_inv(self, x, y, theta, cx, cy, ego_px):
        """Global pixel → ego pixel affine (3×3, row-major)."""
        ct, st = math.cos(theta), math.sin(theta)
        r = self._gm_px / ego_px
        gx = self._gm_ox + x / self._gm_px
        gy = self._gm_oy - y / self._gm_px
        return np.float32([
            [ r * ct, -r * st, cx - r * ct * gx + r * st * gy],
            [ r * st,  r * ct, cy - r * st * gx - r * ct * gy],
            [0, 0, 1]])

    def _gmap_affine_fwd(self, x, y, theta, cx, cy, ego_px):
        """Ego pixel → global pixel affine (3×3, row-major)."""
        ct, st = math.cos(theta), math.sin(theta)
        s = ego_px / self._gm_px
        return np.float32([
            [ s * ct,  s * st,
              self._gm_ox + x / self._gm_px - cx * s * ct - cy * s * st],
            [-s * st,  s * ct,
              self._gm_oy - y / self._gm_px + cx * s * st - cy * s * ct],
            [0, 0, 1]])

    def gmap_update_gpu(self, obs_ego, known_ego, x, y, theta,
                        cx, cy, ego_px):
        """GPU evidence update. Returns True on success."""
        if not getattr(self, '_gm_configured', False):
            return False
        if not self._gl_ready:
            return False
        if not self._gm_gl_ready:
            try:
                self._init_gmap_gl()
            except Exception as e:
                print("gpu_render: gmap init failed: %s" % e)
                import traceback; traceback.print_exc()
                return False

        t0 = time.monotonic()

        h, w = obs_ego.shape[:2]
        ego_enc = np.zeros((h, w), dtype=np.uint8)
        free = (known_ego > 0) & (obs_ego == 0)
        ego_enc[free] = 1
        obs_px = obs_ego > 0
        ego_enc[obs_px] = np.minimum(
            obs_ego[obs_px].astype(np.uint16) + 1, 101).astype(np.uint8)

        self._gm_ego_tex.write(ego_enc.tobytes())

        M_inv = self._gmap_affine_inv(x, y, theta, cx, cy, ego_px)

        read_idx = self._gm_idx
        write_idx = 1 - read_idx

        self._gm_conf[read_idx].use(location=0)
        self._gm_hmap[read_idx].use(location=1)
        self._gm_ego_tex.use(location=2)
        self._gm_prog_up['u_inv'].write(M_inv.T.tobytes())

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._gm_fbo[write_idx].use()
        self._gm_vao_up.render(moderngl.TRIANGLE_STRIP)

        self._gm_idx = write_idx
        self._gm_n += 1
        t1 = time.monotonic()

        if self._gm_n <= 3 or self._gm_n % 300 == 0:
            data = self._gm_conf[self._gm_idx].read()
            cmap = np.frombuffer(data, dtype=np.uint8)
            nfree = int(np.count_nonzero(cmap > 190))
            nobs = int(np.count_nonzero(cmap < 90))
            ntot = self._gm_mw * self._gm_mh
            print("gpu_gmap: free=%d obs=%d unk=%d | "
                  "pose=(%.3f,%.3f,%.1f°) %.1fms"
                  % (nfree, nobs, ntot - nfree - nobs,
                     x, y, math.degrees(theta), (t1 - t0) * 1e3))

        return True

    def gmap_project_gpu(self, x, y, theta, cx, cy, ego_px, eh, ew):
        """Project global conf map to ego space on GPU. Returns uint8 (eh, ew)."""
        if not self._gm_gl_ready:
            return None

        M_fwd = self._gmap_affine_fwd(x, y, theta, cx, cy, ego_px)

        self._gm_conf[self._gm_idx].use(location=0)
        self._gm_prog_pj['u_fwd'].write(M_fwd.T.tobytes())

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._gm_proj_fbo.use()
        self._gm_vao_pj.render(moderngl.TRIANGLE_STRIP)

        data = self._gm_proj_fbo.read(components=1, alignment=1)
        return np.frombuffer(data, dtype=np.uint8).reshape(eh, ew).copy()

    def gmap_reset(self):
        """Reset GPU map textures to unknown (for SLAM rebuild)."""
        if not self._gm_gl_ready:
            return
        init_conf = np.full(self._gm_mw * self._gm_mh, 128, dtype=np.uint8)
        init_hmap = np.zeros(self._gm_mw * self._gm_mh, dtype=np.uint8)
        for t in self._gm_conf:
            t.write(init_conf.tobytes())
        for t in self._gm_hmap:
            t.write(init_hmap.tobytes())
        self._gm_n = 0

    # ── Blob smooth pass ────────────────────────────────────────

    def _update_blob(self):
        """4-pass blur: max-dilate H/V then Gaussian H/V → self._blob_b_tex."""
        if not getattr(self, '_gm_gl_ready', False):
            return
        hmap = self._gm_hmap[self._gm_idx]
        conf = self._gm_conf[self._gm_idx]

        passes = [
            (hmap, self._blob_a_fbo, 0),
            (self._blob_a_tex, self._blob_b_fbo, 1),
            (self._blob_b_tex, self._blob_a_fbo, 2),
            (self._blob_a_tex, self._blob_b_fbo, 3),
        ]

        self._ctx.disable(moderngl.DEPTH_TEST)
        for src_tex, dst_fbo, mode in passes:
            dst_fbo.use()
            src_tex.use(location=0)
            conf.use(location=1)
            self._prog_bblur['u_mode'].value = mode
            self._vao_bblur.render(moderngl.TRIANGLE_STRIP)

    # ── Per-frame render ─────────────────────────────────────────

    def render(self, x, y, theta, cameras=None,
               trail_xy=None, fwd_scale=1.0, bwd_scale=1.0, ang_scale=1.0,
               battery_frac=0.0):
        """Render the full atlas (3D + cameras + HUD). Returns (vh, vw, 3) uint8."""
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

        if self._pbo_ready:
            prev_pbo = self._pbo[1 - self._pbo_idx]
            data = prev_pbo.read()
            self._out[:] = np.frombuffer(data, dtype=np.uint8).reshape(
                self._vh, self._vw, 3)[::-1]

        ctx = self._ctx
        v3h = self._view3d_h
        ct, st = math.cos(theta), math.sin(theta)

        # ── Pre-compute blob heightmap (runs in blob FBOs) ──
        self._update_blob()

        # ── 3D scene → scene FBO (for SSAO) ──
        self._scene_fbo.use()
        self._scene_fbo.clear(0.0, 0.0, 0.0, 1.0)
        ctx.viewport = (0, 0, self._vw, v3h)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<'

        aspect = self._vw / v3h
        if self.topdown:
            cam = np.float32([x, 15.0, -y])
            tgt = np.float32([x, 0.0, -y])
            up = np.float32([ct, 0, -st])
            half_ext = 6.0
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
            proj = _perspective(CAM_FOV_DEG, aspect, NEAR_CLIP, FAR_CLIP)
            self._prog_t['u_topdown'].value = 0

        mvp = proj @ view

        R0 = np.float32([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        Ry = np.float32([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        R_robot = Ry @ R0
        model = np.eye(4, dtype=np.float32)
        model[:3, :3] = R_robot
        model[:3, 3] = [x, 0, -y]
        mvp_robot = proj @ view @ model

        self._blob_b_tex.use(location=0)
        self._gm_conf[self._gm_idx].use(location=1)
        self._prog_t['u_mvp'].write(mvp.T.astype(np.float32).tobytes())
        self._prog_t['u_cam'].value = tuple(cam.tolist())
        self._vao_t.render()

        if self._vao_r is not None:
            self._prog_r['u_mvp'].write(
                mvp_robot.T.astype(np.float32).tobytes())
            self._prog_r['u_nmat'].write(
                R_robot.T.astype(np.float32).tobytes())
            self._vao_r.render()

        if trail_xy is not None and len(trail_xy) >= 2:
            self._render_trail(trail_xy, mvp)

        # ── SSAO + proximity composite → main FBO ──
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0, 1.0)
        ctx.viewport = (0, 0, self._vw, v3h)
        ctx.disable(moderngl.DEPTH_TEST)
        self._scene_color_tex.use(location=0)
        self._scene_depth_tex.use(location=1)
        ssao = self._prog_ssao
        ssao['u_intensity'].value = (
            0.0 if self.topdown else SSAO_INTENSITY)
        inv_vp = np.linalg.inv(mvp).T.astype(np.float32)
        ssao['u_inv_vp'].write(inv_vp.tobytes())
        ssao['u_robot_pos'].value = (float(x), 0.0, float(-y))
        ssao['u_heading'].value = float(theta)
        ssao['u_fwd_scale'].value = float(fwd_scale)
        ssao['u_bwd_scale'].value = float(bwd_scale)
        ssao['u_ang_scale'].value = float(ang_scale)
        self._vao_ssao.render(moderngl.TRIANGLE_STRIP)

        # ── Switch to 2D overlay mode (full atlas viewport, no depth) ──
        ctx.viewport = (0, 0, self._vw, self._vh)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ── Minimap (overlaid on 3D view, upper-left of 3D area) ──
        mm_x0 = MINIMAP_PAD
        mm_y0 = v3h - MINIMAP_PAD - MINIMAP_SZ
        ctx.viewport = (mm_x0, mm_y0, MINIMAP_SZ, MINIMAP_SZ)
        self._gm_conf[self._gm_idx].use(location=0)
        self._gm_hmap[self._gm_idx].use(location=1)
        mm = self._prog_mm
        mm['u_conf'].value = 0
        mm['u_hmap'].value = 1
        mm['u_vp'].value = (float(mm_x0), float(mm_y0),
                            float(MINIMAP_SZ), float(MINIMAP_SZ))
        map_cx = float(self._mw) / 2.0 + x / PX_SIZE
        map_cy = float(self._mh) / 2.0 - y / PX_SIZE
        mm['u_center'].value = (map_cx, map_cy)
        mm['u_zoom'].value = MINIMAP_ZOOM
        mm['u_mapsz'].value = (float(self._mw), float(self._mh))
        sf = min(fwd_scale, bwd_scale, ang_scale)
        mm['u_sf'].value = sf
        mm['u_theta'].value = float(theta)
        self._vao_mm.render(moderngl.TRIANGLE_STRIP)

        # ── Camera feeds (top strip of atlas) ──
        ctx.disable(moderngl.BLEND)
        cam_y0_gl = self._vh - self.CAM_H
        if cameras is not None:
            for i, img in enumerate(cameras):
                if img is None:
                    continue
                self._cam_textures[i].write(img.tobytes())
                self._cam_textures[i].use(location=0)
                vp_x = i * self.CAM_W
                ctx.viewport = (vp_x, cam_y0_gl, self.CAM_W, self.CAM_H)
                self._prog_cam['u_tex'].value = 0
                self._prog_cam['u_vp'].value = (
                    float(vp_x), float(cam_y0_gl),
                    float(self.CAM_W), float(self.CAM_H))
                self._vao_cam.render(moderngl.TRIANGLE_STRIP)

        # ── Battery bar (bottom of camera strip) ──
        bar_y_gl = cam_y0_gl
        ctx.viewport = (0, bar_y_gl, self._vw, self.BAR_H)
        self._prog_bat['u_vp'].value = (0.0, float(bar_y_gl),
                                         float(self._vw), float(self.BAR_H))
        self._prog_bat['u_frac'].value = float(battery_frac)
        self._vao_bat.render(moderngl.TRIANGLE_STRIP)

        # ── Async PBO readback ──
        ctx.viewport = (0, 0, self._vw, self._vh)
        ctx.disable(moderngl.BLEND)
        cur_pbo = self._pbo[self._pbo_idx]
        self._fbo.read_into(cur_pbo, components=3, alignment=1)

        if not self._pbo_ready:
            data = cur_pbo.read()
            self._out[:] = np.frombuffer(data, dtype=np.uint8).reshape(
                self._vh, self._vw, 3)[::-1]
            self._pbo_ready = True

        self._pbo_idx = 1 - self._pbo_idx

        if not hasattr(self, '_rn'):
            self._rn = 0
        self._rn += 1
        if self._rn <= 3 or self._rn % 100 == 0:
            print("gpu_render: %.1fms" % ((time.monotonic() - t0) * 1e3))

        return self._out

    def _render_trail(self, trail_xy, mvp):
        """Render trail as GL line strip in the 3D viewport."""
        n = min(len(trail_xy), 300)
        pts = np.empty((n, 3), dtype=np.float32)
        pts[:, 0] = trail_xy[-n:, 0]
        pts[:, 1] = 0.02
        pts[:, 2] = -trail_xy[-n:, 1]
        self._trail_vbo.orphan(n * 3 * 4)
        self._trail_vbo.write(pts.tobytes())
        self._prog_trail['u_mvp'].write(mvp.T.astype(np.float32).tobytes())
        self._vao_trail.render(moderngl.LINE_STRIP, vertices=n)

    # ── Cleanup ──────────────────────────────────────────────────

    def release(self):
        if hasattr(self, '_ctx'):
            try:
                self._ctx.release()
            except Exception:
                pass
