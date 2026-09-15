"""Simulated people for Kevin: floor circles + chat if they're in the mood."""
from __future__ import annotations

import math
import random


HELLO = (
    "hey! love this energy :)",
    "hi Kevin! you look busy in a good way",
    "morning! casita feels alive with you around",
    "oh hey little guy — nice pathing",
)
BYE = (
    "catch you later!",
    "go wander, I'm good",
    "bye Kevin!",
)
BUSY = (
    "not now — in the zone",
    "head down, maybe later",
    "mm, give me a bit",
)


class SimPeople:
    def __init__(self, seed=3):
        rng = random.Random(seed + 11)
        # Room interiors, off door columns and off spawn.
        spots = (
            ("Ana", (-2.90, -5.55)),
            ("Marco", (2.15, -5.35)),
            ("Priya", (0.35, 0.45)),
            ("Jules", (2.05, 4.05)),
        )
        self.folk = []
        for name, xy in spots:
            self.folk.append({
                "name": name,
                "xy": xy,
                "r": 0.30,
                "mood": rng.random() > 0.35,
                "cool": 0.0,
                "chat_left": 0.0,
                "said": "",
            })
        self.active = None
        self.bubble = ""

    def as_keepout(self):
        return [
            {"cx": p["xy"][0], "cy": p["xy"][1], "yaw": 0.0, "hx": 0.35, "hy": 0.35, "hz": 0.9}
            for p in self.folk
        ]

    def nearest(self, xy):
        best = None
        best_d = 1e9
        for p in self.folk:
            d = math.hypot(xy[0] - p["xy"][0], xy[1] - p["xy"][1])
            if d < best_d:
                best_d = d
                best = p
        return best, best_d

    def tick(self, xy, yaw, dt, rng=None):
        """Return (v_hold, bubble). v_hold True means stop for a chat."""
        rng = rng or random
        x, y = float(xy[0]), float(xy[1])
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        self.bubble = ""
        for p in self.folk:
            p["cool"] = max(0.0, float(p["cool"]) - float(dt))
            if p["chat_left"] > 0.0:
                p["chat_left"] = max(0.0, float(p["chat_left"]) - float(dt))
                self.bubble = p["said"]
                self.active = p["name"]
                if p["chat_left"] <= 0.0:
                    p["said"] = "%s: %s" % (p["name"], rng.choice(BYE))
                    self.bubble = p["said"]
                    p["cool"] = 22.0
                    self.active = None
                return True, self.bubble

        p, dist = self.nearest((x, y))
        if p is None:
            return False, ""
        fx = p["xy"][0] - x
        fy = p["xy"][1] - y
        ahead = (fx * c + fy * s) / max(dist, 1e-6)
        see = dist < 1.65 and ahead > 0.25
        if not see or p["cool"] > 0.0:
            return False, ""
        if p["mood"]:
            p["said"] = "%s: %s" % (p["name"], rng.choice(HELLO))
            p["chat_left"] = rng.uniform(2.2, 3.6)
            self.bubble = p["said"]
            self.active = p["name"]
            print("chat start", p["name"], "mood=yes dist=%.2f" % dist, flush=True)
            return True, self.bubble
        p["said"] = "%s: %s" % (p["name"], rng.choice(BUSY))
        p["cool"] = 18.0
        self.bubble = p["said"]
        print("chat skip", p["name"], "mood=no dist=%.2f" % dist, flush=True)
        return False, self.bubble
