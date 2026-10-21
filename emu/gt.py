"""Ground-truth / scenario control client for emu.world (eval tools only; robot code never imports this)."""
import json, math, socket
CTL = ("127.0.0.1", 47801)
_s = None
def ctl(cmd, **kw):
    global _s
    if _s is None: _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); _s.settimeout(1.0)
    _s.sendto(json.dumps(dict(cmd=cmd, **kw)).encode(), CTL); return json.loads(_s.recv(65536))
def state(): return ctl("state")
def to_map(p, start):
    """world (x,y,yaw) -> robot map frame (= start pose of the run)."""
    dx, dy = p[0] - start[0], p[1] - start[1]; c, s = math.cos(start[2]), math.sin(start[2])
    return (c * dx + s * dy, -s * dx + c * dy, (p[2] - start[2] + math.pi) % (2 * math.pi) - math.pi)
def to_world(p, start):
    c, s = math.cos(start[2]), math.sin(start[2])
    return (start[0] + c * p[0] - s * p[1], start[1] + s * p[0] + c * p[1])
