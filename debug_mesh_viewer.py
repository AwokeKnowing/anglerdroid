"""Interactive 3D robot mesh viewer — run on desktop to inspect the mesh.

Usage:  python debug_mesh_viewer.py
  - Drag to rotate
  - Scroll to zoom
  - Close window when done
"""
import numpy as np
import cv2
import sys

R = 0.0857  # wheel radius

def build_robot_mesh():
    vl, fl, cl = [], [], []

    def _box(x0, y0, z0, x1, y1, z1, rgb):
        n = len(vl)
        vl.extend([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                    [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        for q in ([n,n+3,n+2,n+1],[n+4,n+5,n+6,n+7],
                   [n,n+1,n+5,n+4],[n+2,n+3,n+7,n+6],
                   [n,n+4,n+7,n+3],[n+1,n+2,n+6,n+5]):
            fl.append(np.array(q, dtype=np.int32))
            cl.append(rgb)

    def _wheel(cx, cz, y0, y1, radius, rgb, ns=10):
        n = len(vl)
        ang = np.linspace(0, 2*np.pi, ns, endpoint=False)
        for a in ang:
            vl.append([cx+radius*np.cos(a), y0, cz+radius*np.sin(a)])
        for a in ang:
            vl.append([cx+radius*np.cos(a), y1, cz+radius*np.sin(a)])
        for i in range(ns):
            j = (i+1) % ns
            fl.append(np.array([n+i, n+ns+i, n+ns+j, n+j], dtype=np.int32))
            cl.append(rgb)
        fl.append(np.arange(n, n+ns, dtype=np.int32))
        cl.append([min(c+12, 255) for c in rgb])
        fl.append(np.arange(n+2*ns-1, n+ns-1, -1, dtype=np.int32))
        cl.append([min(c+12, 255) for c in rgb])

    z_base = 0.05
    z_top  = 0.22
    x_back = -0.15     # 15 cm behind wheel axle (box portion)
    x_front = 0.13     # 13 cm in front of wheel (slope portion)
    x_slope = 0.0      # slope starts at wheel axle
    y_hw = 0.16

    n = len(vl)
    vl.extend([
        [x_front, -y_hw, z_base],
        [x_back,  -y_hw, z_base],
        [x_back,  -y_hw, z_top],
        [x_slope, -y_hw, z_top],
        [x_front,  y_hw, z_base],
        [x_back,   y_hw, z_base],
        [x_back,   y_hw, z_top],
        [x_slope,  y_hw, z_top],
    ])
    body_rgb = [45, 45, 50]
    for q in ([n,n+1,n+5,n+4],
              [n+1,n+2,n+6,n+5],
              [n+2,n+3,n+7,n+6],
              [n+3,n,n+4,n+7],
              [n,n+3,n+2,n+1],
              [n+4,n+5,n+6,n+7]):
        fl.append(np.array(q, dtype=np.int32))
        cl.append(body_rgb)

    _wheel(0, R, -0.21, -0.16, R*0.97, [22, 22, 26])
    _wheel(0, R,  0.16,  0.21, R*0.97, [22, 22, 26])
    _box(-0.20, -0.015, 0.0, -0.16, 0.015, 0.04, [40, 40, 45])     # caster at back
    _box(-0.165, -0.015, z_top, -0.135, 0.015, 1.00, [100, 105, 115])  # mast from back wall
    _box(-0.135, -0.015, 0.97, 0.015, 0.015, 1.00, [85, 90, 100])  # camera arm forward
    _box(-0.05, -0.04, 1.00, 0.04, 0.04, 1.025, [30, 55, 80])     # RS1 top-down
    _box(0.015, -0.03, 0.94, 0.05, 0.03, 0.97, [30, 55, 80])      # RS2 forward
    _box(-0.14, -0.04, z_top, -0.04, 0.04, z_top + 0.025, [40, 55, 40])  # Jetson NX

    # ground plane quad for reference
    g = 1.0
    gn = len(vl)
    vl.extend([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]])
    fl.append(np.array([gn, gn+3, gn+2, gn+1], dtype=np.int32))
    cl.append([180, 175, 165])

    return np.array(vl, dtype=np.float32), fl, np.array(cl, dtype=np.uint8)


def render_view(verts, faces, colors, cam_pos, look_at, up_hint, w, h):
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)
    rt = np.cross(fwd, up_hint)
    rn = np.linalg.norm(rt)
    rt = rt / rn if rn > 1e-6 else np.float32([1, 0, 0])
    up_a = np.cross(rt, fwd)
    Rm = np.stack([rt, up_a, -fwd], axis=1).astype(np.float32)
    cv3 = (verts - cam_pos) @ Rm
    f = min(w, h) * 1.3
    ok = cv3[:, 2] < -0.01
    sx = np.where(ok, cv3[:, 0] * f / (-cv3[:, 2]) + w * 0.5, -9999)
    sy = np.where(ok, -cv3[:, 1] * f / (-cv3[:, 2]) + h * 0.5, -9999)
    light = np.float32([0.2, -0.3, -0.9])
    light /= np.linalg.norm(light)
    draw = []
    for i, fidx in enumerate(faces):
        fv = cv3[fidx]
        if (fv[:, 2] >= -0.01).any():
            continue
        e1, e2 = fv[1] - fv[0], fv[-1] - fv[0]
        nrm = np.cross(e1, e2)
        nl = np.linalg.norm(nrm)
        if nl < 1e-8:
            continue
        nrm /= nl
        if np.dot(nrm, fv.mean(axis=0)) > 0:
            continue
        brt = max(0.3, min(1.0, -np.dot(nrm, light)))
        pts = np.column_stack([sx[fidx].astype(np.int32),
                               sy[fidx].astype(np.int32)])
        col = np.clip(colors[i].astype(np.float32) * brt,
                      0, 255).astype(np.uint8).tolist()
        draw.append((fv[:, 2].mean(), pts, col))
    draw.sort(key=lambda d: d[0])
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    for _, pts, col in draw:
        cv2.fillConvexPoly(img, pts, col)
    for _, pts, col in draw:
        cv2.polylines(img, [pts], True,
                      [max(0, c - 20) for c in col], 1, cv2.LINE_AA)
    return img


W, H = 800, 600

verts, faces, colors = build_robot_mesh()
center = np.float32([0.08, 0, 0.15])

azimuth = 210.0   # degrees (start from behind-right)
elevation = 30.0   # degrees above horizon
distance = 2.0

drag_start = None
drag_az0 = 0.0
drag_el0 = 0.0

def cam_from_angles():
    az = np.radians(azimuth)
    el = np.radians(elevation)
    x = distance * np.cos(el) * np.cos(az)
    y = distance * np.cos(el) * np.sin(az)
    z = distance * np.sin(el)
    return center + np.float32([x, y, z])

def mouse_cb(event, mx, my, flags, param):
    global drag_start, drag_az0, drag_el0, azimuth, elevation, distance
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (mx, my)
        drag_az0 = azimuth
        drag_el0 = elevation
    elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
        dx = mx - drag_start[0]
        dy = my - drag_start[1]
        azimuth = drag_az0 - dx * 0.3
        elevation = np.clip(drag_el0 + dy * 0.3, -89, 89)
    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            distance = max(0.3, distance * 0.9)
        else:
            distance = min(10.0, distance * 1.1)

cv2.namedWindow("Robot Mesh Viewer", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Robot Mesh Viewer", mouse_cb)

presets = {
    ord('1'): ("Side (right)", 0, 0, 1.5),
    ord('2'): ("Top", 0, 89, 2.0),
    ord('3'): ("Back", 180, 0, 1.5),
    ord('4'): ("Drone (follow-cam)", 210, 35, 2.5),
    ord('5'): ("Front", 0, 5, 1.5),
    ord('6'): ("Side (left)", 180, 0, 1.5),
}

print("Robot Mesh Viewer")
print("  Drag to rotate, scroll to zoom")
print("  Keys: 1=Side  2=Top  3=Back  4=Drone  5=Front  6=Left  q=Quit")

while True:
    cam_pos = cam_from_angles()
    img = render_view(verts, faces, colors, cam_pos, center,
                      np.float32([0, 0, 1]), W, H)
    info = "az=%.0f  el=%.0f  d=%.1f" % (azimuth % 360, elevation, distance)
    cv2.putText(img, info, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "+X=fwd  +Y=left  +Z=up", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.imshow("Robot Mesh Viewer", img)
    k = cv2.waitKey(16) & 0xFF
    if k == ord('q') or k == 27:
        break
    if k in presets:
        name, azimuth, elevation, distance = presets[k]
        print("  → %s view" % name)

cv2.destroyAllWindows()
