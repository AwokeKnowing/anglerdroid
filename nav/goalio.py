"""Goal interface between goal PRODUCERS (LightNav process, scripted code+labels, RL policy, ...) and the nav core loop.
Flat shm blocks (emu.shm.Block seqlock, CLOCK_MONOTONIC stamps).  Layouts are the contract; see docs/ARCHITECTURE_LOOPS.md.

  kevin_navstate        nav -> everyone, every nav tick (30 Hz)
      odom   f8[3]  x, y, th of the robot in the ODOMETRY frame at t_od
      t_od   f8[1]  monotonic s of that pose
      mode   u1[1]  0 idle 1 path 2 turn 3 drive 4 backout 5 other
      ack    u8[1]  gseq of the last goal nav accepted (0 none)
      map_T  f8[3]  T map<-odom (SLAM layer; identity until localization publishes one)   [reserved, step 4]
      map_c  f4[1]  confidence of map_T in [0,1]                                           [reserved, step 4]
  kevin_task_<src>      orchestrator(nav for now) -> producer: what to pursue
      tseq   u8[1]  bumps on every new task / cancel
      active u1[1]  1 = pursue text, 0 = cancelled
      text   u1[256] UTF-8, NUL padded
  kevin_goal_<src>      producer -> nav: the goal it wants now
      gseq   u8[1]  bumps on every new goal (nav acts on a change)
      source u1[1]  SOURCES id
      kind   u1[1]  1 pose (x,y,th; th NaN = any heading)  2 face (th)  3 stop (task done / cancel motion)
      pose   f8[3]  ODOMETRY frame
      vmax   f4[1]  m/s cap requested by the producer (0 = nav default)     [recorded, not enforced yet]
      ttl    f4[1]  s the goal stays valid without a refresh (0 = until replaced) [recorded, not enforced yet]
      tseq   u8[1]  task this goal belongs to (stale-task goals are ignored)
"""
from __future__ import annotations
import math, time
import numpy as np
from emu import shm

SOURCES = {"lightnav": 1, "script": 2, "rl": 3}
MODES = {"idle": 0, "path": 1, "turn": 2, "drive": 3, "backout": 4}
L_NAVSTATE = [("odom", (3,), "<f8"), ("t_od", (1,), "<f8"), ("mode", (1,), "u1"), ("ack", (1,), "<u8"), ("map_T", (3,), "<f8"), ("map_c", (1,), "<f4")]
L_TASK = [("tseq", (1,), "<u8"), ("active", (1,), "u1"), ("text", (256,), "u1")]
L_GOAL = [("gseq", (1,), "<u8"), ("source", (1,), "u1"), ("kind", (1,), "u1"), ("pose", (3,), "<f8"), ("vmax", (1,), "<f4"), ("ttl", (1,), "<f4"), ("tseq", (1,), "<u8")]
K_POSE, K_FACE, K_STOP = 1, 2, 3


def open_block(name, layout, wait_s=10.0):
    t_end = time.monotonic() + wait_s
    while True:
        try: return shm.Block(name, layout)
        except FileNotFoundError:
            if time.monotonic() > t_end: raise
            time.sleep(0.1)


def enc_text(s):
    b = np.zeros(256, np.uint8); r = s.encode()[:255]; b[:len(r)] = np.frombuffer(r, np.uint8); return b


def dec_text(a):
    return bytes(a[: int(np.argmax(a == 0)) if (a == 0).any() else len(a)]).decode(errors="replace")


class GoalPort:
    """nav side of one producer: owns kevin_task_<src> (writer) and reads kevin_goal_<src>.  Also quacks like the old
    in-process VLN object (instruct/cancel/state/text) so nav's command paths stay unchanged."""
    def __init__(self, src, log):
        self.src = src; self.log = log; self.text = None; self.tseq = 0; self.last_gseq = 0
        self.task = shm.Block("kevin_task_" + src, L_TASK, create=True); self.goal = None; self.n_goals = 0; self._stat = None
        self.task.write(time.monotonic_ns(), tseq=0, active=0, text=enc_text(""))

    def instruct(self, text):
        self.tseq += 1; self.text = text; self.task.write(time.monotonic_ns(), tseq=self.tseq, active=1, text=enc_text(text)); self.log("VLN: '%s'" % text)

    def cancel(self):
        if self.text is None: return
        self.tseq += 1; self.text = None; self.task.write(time.monotonic_ns(), tseq=self.tseq, active=0, text=enc_text(""))

    def poll(self):
        """-> list of nav command dicts for a NEW goal (odom frame, tagged frame='odom'), else []."""
        if self.goal is None:
            try: self.goal = shm.Block("kevin_goal_" + self.src, L_GOAL)
            except FileNotFoundError: return []
        if self.goal.seq == self.last_gseq: return []
        s, st, g = self.goal.read(); gseq = int(g["gseq"][0])
        self.last_gseq = s
        if gseq == 0 or int(g["tseq"][0]) != self.tseq or self.text is None: return []     # goal for an old/cancelled task
        self.n_goals += 1; k = int(g["kind"][0]); x, y, th = (float(v) for v in g["pose"])
        if k == K_STOP: self.text = None; self.log("VLN: model says stop"); return [dict(kind="stop")]
        if k == K_FACE: return [dict(kind="face", rad=th, frame="odom")]
        return [dict(kind="goto_xy", x=x, y=y, heading=None if math.isnan(th) else th, frame="odom")]

    def state(self):
        st = dict(url="shm:kevin_goal_" + self.src, instruction=self.text, active=self.text is not None, goals=self.n_goals, tseq=self.tseq)
        try:
            if self._stat is None: self._stat = shm.Block("kevin_stat_" + self.src, L_STAT)
            _, _, a = self._stat.read(); st.update({k: float(v[0]) for k, v in a.items()})
        except Exception: pass
        return st


# producer health (producer -> anyone): fixed-period loop stats
L_STAT = [("ticks", (1,), "<f8"), ("late_ms_max", (1,), "<f4"), ("late_ms_p99", (1,), "<f4"), ("period_ms", (1,), "<f4"), ("pred_ms", (1,), "<f4"),
          ("preds", (1,), "<f8"), ("errors", (1,), "<f8"), ("fps", (1,), "<f4")]
