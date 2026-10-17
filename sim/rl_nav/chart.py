#!/usr/bin/env python3
"""Rewrite the live rl_nav canvas from train.log / metrics.jsonl."""
from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
LOG = OUT / "train.log"
JSONL = OUT / "metrics.jsonl"
CANVAS = Path(
    "/home/james/.cursor/projects/home-james-gits-awokeknowing-anglerdroid/canvases/rl-nav-live.canvas.tsx"
)
CLIP_CANVAS = Path(
    "/home/james/.cursor/projects/home-james-gits-awokeknowing-anglerdroid/canvases/rl-nav-clip.canvas.tsx"
)
_CLIP_STAMP = 0.0

POOL_SEC_RE = re.compile(r"house pool n=\d+.*?sec=([\d.]+)")


def pool_ceil(path: Path = LOG) -> float | None:
    """Deprecated house-gen fog count (sec=). Do not use as reward ceiling."""
    if not path.is_file():
        return None
    first = None
    for line in path.read_text(errors="replace").splitlines():
        m = POOL_SEC_RE.search(line)
        if m:
            first = float(m.group(1))
            break
    return first


UPD_RE = re.compile(
    r"upd\s+(\d+)\s+step=(\d+)\s+sps=([\d.]+)\s+ent=([\d.]+)"
    r".*?hits=([\d.]+).*?crash=([\d.]+).*?clear=([\d.]+).*?stg=(\d+)"
    r"(?:.*?loss=([-\d.]+))?(?:.*?ret=([-\d.]+))?(?:.*?ceil=([\d.]+))?"
    r"(?:.*?(?:rewards|cover)=([\d.]+))?(?:.*?contact=([-\d.]+))?(?:.*?cfrac=([\d.]+))?"
    r"(?:.*?stuck_s=([-\d.]+))?"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = UPD_RE.search(line)
        if not m:
            continue
        stg = int(m.group(8))
        rec = {
            "upd": int(m.group(1)),
            "step": int(m.group(2)),
            "sps": float(m.group(3)),
            "ent": float(m.group(4)),
            "hits": float(m.group(5)),
            "crash": float(m.group(6)),
            "clear": float(m.group(7)),
            "stg": stg,
        }
        if m.group(11):
            rec["ceil"] = float(m.group(11))
            rec["ret_ceil"] = rec["ceil"] + 1.0
        else:
            rec["ceil"] = None
            rec["ret_ceil"] = None
        if m.group(9):
            rec["loss"] = float(m.group(9))
        if m.group(10):
            rec["ret"] = float(m.group(10))
        if m.group(12):
            rec["rewards"] = rec["cover"] = float(m.group(12))
        if m.group(13):
            rec["contact"] = float(m.group(13))
        if m.group(14):
            rec["contact_frac"] = float(m.group(14))
        if m.lastindex and m.lastindex >= 15 and m.group(15):
            rec["stuck_s"] = float(m.group(15))
        elif rec.get("contact") is not None:
            rec["stuck_s"] = float(rec["contact"]) * 0.20
        rows.append(rec)
    return rows


def parse_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        stg = int(rec.get("stg", rec.get("stage", 0)))
        rec["stg"] = stg
        if rec.get("ceil") is None:
            rec.pop("ceil", None)
        else:
            rec["ceil"] = float(rec["ceil"])
            rec.setdefault("ret_ceil", rec["ceil"] + 1.0)
        if "rewards" not in rec and "cover" in rec:
            rec["rewards"] = rec["cover"]
        elif "cover" not in rec and "rewards" in rec:
            rec["cover"] = rec["rewards"]
        rows.append(rec)
    return rows


def merge_rows(log_rows: list[dict], json_rows: list[dict]) -> list[dict]:
    by_step = {}
    for r in log_rows + json_rows:
        k = int(r.get("step", 0))
        if k <= 0:
            continue
        prev = by_step.get(k, {})
        prev.update(r)
        by_step[k] = prev
    return [by_step[k] for k in sorted(by_step)]


def downsample(rows: list[dict], n: int = 40) -> list[dict]:
    if len(rows) <= n:
        return rows
    if n <= 2:
        return rows[:1] + rows[-1:]
    out = [rows[0]]
    last = len(rows) - 1
    for i in range(1, n - 1):
        idx = 1 + int(round(i * (last - 1) / float(n - 1)))
        idx = min(last - 1, max(1, idx))
        if rows[idx] is not out[-1]:
            out.append(rows[idx])
    if rows[-1] is not out[-1]:
        out.append(rows[-1])
    return out


def _fmt(xs: list[float], nd=3) -> str:
    return "[" + ", ".join(f"{x:.{nd}f}" for x in xs) + "]"


def _cats(rows: list[dict]) -> str:
    labs = []
    for r in rows:
        step = int(r.get("step", 0))
        if step >= 1_000_000:
            labs.append(f"{step / 1_000_000:.1f}M")
        else:
            labs.append(f"{step / 1000:.0f}k")
    return "[" + ", ".join(json.dumps(s) for s in labs) + "]"


def render_canvas(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("no rows")
    pts = downsample(rows, 40)
    last = rows[-1]
    ceil = float(last.get("ceil") or 1.0)
    hits = float(last.get("hits", 0.0) or 0.0)
    rew = float(last.get("rewards", last.get("cover", 0.0)) or 0.0)
    crash = float(last.get("crash", 0.0) or 0.0)
    ent = float(last.get("ent", 0.0) or 0.0)
    step = int(last.get("step", 0))
    sps = float(last.get("sps", 0.0) or 0.0)
    stg = int(last.get("stg", 0))
    loss_pts = downsample([r for r in rows if "loss" in r], 40)
    ret_pts = downsample([r for r in rows if "ret" in r], 40)
    cats = _cats(pts)

    def _c(r):
        v = r.get("ceil")
        return float(v) if v is not None else ceil

    hits_s = _fmt([float(r.get("hits") or 0.0) for r in pts])
    ceil_s = _fmt([_c(r) for r in pts])
    rew_s = _fmt([float(r.get("rewards", r.get("cover", 0.0)) or 0.0) for r in pts])
    crash_s = _fmt([float(r.get("crash") or 0.0) for r in pts])
    cfrac_s = _fmt([float(r.get("contact_frac", 0.0) or 0.0) for r in pts])

    def _stuck(r):
        if r.get("stuck_s") is not None:
            return float(r["stuck_s"])
        if r.get("contact") is not None:
            return float(r["contact"]) * 0.20
        return 0.0

    stuck_s = _fmt([_stuck(r) for r in pts])
    stuck_last = _stuck(last)
    cfrac_last = float(last.get("contact_frac", 0.0) or 0.0)
    stuck_ymax = max(8.0, max((_stuck(r) for r in pts), default=0.0) * 1.25)

    loss_block = ""
    if len(loss_pts) >= 2:
        loss_block = f"""
      <Stack gap={{8}}>
        <H2>PPO loss</H2>
        <Text size="small" tone="secondary">
          Clipped surrogate + value + entropy. Lower is not always better if
          rewards / episode are not rising.
        </Text>
        <LineChart
          categories={{{_cats(loss_pts)}}}
          series={{[{{ name: "PPO loss", data: {_fmt([float(r["loss"]) for r in loss_pts])}, tone: "warning" }}]}}
          height={{200}}
          beginAtZero={{false}}
        />
      </Stack>"""

    ret_block = ""
    if len(ret_pts) >= 2:
        ret_ceil_s = _fmt([float(r.get("ceil") or ceil) for r in ret_pts])
        ret_block = f"""
      <Stack gap={{8}}>
        <H2>Episode return</H2>
        <Text size="small" tone="secondary">
          Sum of paint − stall − contact. One claim ≈ 1.0 before penalties.
          Green line is packing budget (open area / 4 m²).
        </Text>
        <LineChart
          categories={{{_cats(ret_pts)}}}
          series={{[
            {{ name: "mean episode return", data: {_fmt([float(r["ret"]) for r in ret_pts])}, tone: "info" }},
            {{ name: "packing ceiling", data: {ret_ceil_s}, tone: "success" }},
          ]}}
          height={{200}}
          beginAtZero={{false}}
        />
      </Stack>"""

    rew_block = f"""
      <Stack gap={{8}}>
        <H2>Reward fraction</H2>
        <Text size="small" tone="secondary">
          Claims / packing ceiling. Curriculum gates 0.32 / 0.45 / 0.55 / 0.65 / 0.72.
          LOS 2 m suppress — a room behind a wall still counts.
        </Text>
        <LineChart
          categories={{{cats}}}
          series={{[
            {{ name: "rewards fraction", data: {rew_s}, tone: "info" }},
          ]}}
          height={{180}}
          beginAtZero
          yMax={{1}}
        />
      </Stack>"""

    ymax = max(ceil, 1.0, max((float(r.get("hits") or 0.0) for r in pts), default=0.0)) * 1.15
    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""import {{
  Callout,
  Grid,
  H1,
  H2,
  LineChart,
  Stack,
  Stat,
  Text,
}} from "cursor/canvas";

export default function RlNavLive() {{
  return (
    <Stack gap={{24}}>
      <Stack gap={{8}}>
        <H1>rl_nav training</H1>
        <Text tone="secondary">
          WanderAround: open-floor waypoints ≥50 cm from walls, 2 m LOS
          suppress (walls keep the far room). Ceiling is area / 4 m²
          (about {ceil:.1f}). Stage {stg}. Canvas rewrites every 60 s.
          Updated {updated}.
        </Text>
      </Stack>

      <Grid columns={{5}} gap={{16}}>
        <Stat value="{hits:.1f} / {ceil:.1f}" label="rewards / episode" />
        <Stat value="{rew:.0%}" label="reward fraction" />
        <Stat value="{stuck_last:.1f}s" label="time in collision / ep" tone="danger" />
        <Stat value="{cfrac_last:.0%}" label="episode time overlapping" tone="warning" />
        <Stat value="{(step / 1e6):.1f}M" label="policy steps" />
      </Grid>

      <Callout tone="info">
        Collision time is hull-overlap seconds per episode (5 Hz steps).
        Target 0 s. Reward ceiling {ceil:.1f} · entropy {ent:.2f} · {sps:.0f} sps.
      </Callout>

      <Stack gap={{8}}>
        <H2>Time in collision</H2>
        <Text size="small" tone="secondary">
          Seconds per episode the rotated hull overlapped a wall (episode
          continues). {cfrac_last:.0%} of episode time. Target is a flat 0.
        </Text>
        <LineChart
          categories={{{cats}}}
          series={{[{{ name: "seconds in collision / episode", data: {stuck_s}, tone: "danger" }}]}}
          height={{220}}
          beginAtZero
          yMax={{{stuck_ymax:.2f}}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Rewards vs packing ceiling</H2>
        <Text size="small" tone="secondary">
          Claims per episode vs open-floor area / (2 m)². LOS only — tiles
          through a wall stay. Not every house can cash the full budget.
        </Text>
        <LineChart
          categories={{{cats}}}
          series={{[
            {{ name: "rewards / episode", data: {hits_s}, tone: "info" }},
            {{ name: "packing ceiling", data: {ceil_s}, tone: "success" }},
          ]}}
          height={{240}}
          beginAtZero
          yMax={{{ymax:.2f}}}
          referenceLines={{[{{ value: {ceil:.2f}, label: "max packing rewards", tone: "success" }}]}}
        />
      </Stack>
{rew_block}
{loss_block}
{ret_block}

      <Stack gap={{8}}>
        <H2>Overlap fraction</H2>
        <Text size="small" tone="secondary">
          Same signal as seconds, as a fraction of policy steps. crash_frac
          is the old cut==2 death rate and stays ~0.
        </Text>
        <LineChart
          categories={{{cats}}}
          series={{[
            {{ name: "fraction of steps in collision", data: {cfrac_s}, tone: "danger" }},
            {{ name: "crash fraction (unused)", data: {crash_s}, tone: "warning" }},
          ]}}
          height={{160}}
          beginAtZero
          yMax={{1}}
        />
      </Stack>
    </Stack>
  );
}}
"""


def _series(rows: list[dict], n: int = 64) -> dict:
    pts = downsample(rows, n) if rows else []

    def nums(key: str) -> list[float | None]:
        out: list[float | None] = []
        for r in pts:
            if key not in r or r[key] is None:
                out.append(None)
            else:
                out.append(float(r[key]))
        return out

    labs = []
    for r in pts:
        step = int(r.get("step", 0))
        labs.append(f"{step / 1e6:.1f}M" if step >= 1_000_000 else f"{step / 1000:.0f}k")

    def pref(*keys: str) -> list[float | None]:
        out: list[float | None] = []
        for r in pts:
            v = None
            for k in keys:
                if k in r and r[k] is not None:
                    v = float(r[k])
                    break
            out.append(v)
        return out

    return {
        "labels": labs,
        "hits": nums("hits"),
        "ceil": nums("ceil"),
        "rewards": pref("rewards", "cover"),
        "cover": pref("rewards", "cover"),
        "crash": nums("crash"),
        "contact": nums("contact"),
        "contact_frac": nums("contact_frac"),
        "stuck_s": nums("stuck_s"),
        "ent": nums("ent"),
        "loss": nums("loss"),
        "ret": nums("ret"),
        "ret_ceil": nums("ret_ceil"),
        "path": nums("path"),
    }


def write_stats(rows: list[dict] | None = None) -> Path:
    if rows is None:
        rows = merge_rows(parse_log(LOG), parse_jsonl(JSONL))
    last = rows[-1] if rows else {}
    meta = {}
    mp = OUT / "watch_meta.json"
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text())
        except json.JSONDecodeError:
            meta = {}
    t_sim = float(meta.get("t_sim", 0) or 0)
    frames = int(meta.get("frames", 0) or 0)
    play_s = (frames / 25.0) if frames > 0 else (
        t_sim if t_sim < 10.0 else 10.0 + (t_sim - 10.0) / 15.0
    )
    try:
        from rl_nav import kernels as _K
        ep_cap = float(_K.EP_S)
    except Exception:
        ep_cap = 1200.0
    pack_max_s = 10.0 + max(0.0, ep_cap - 10.0) / 15.0
    rec = {
        "step": last.get("step", meta.get("step", 0)),
        "sps": last.get("sps", 0),
        "rewards": last.get("rewards", last.get("cover", 0)),
        "cover": last.get("rewards", last.get("cover", 0)),
        "crash": last.get("crash", 0),
        "contact": last.get("contact", 0),
        "contact_frac": last.get("contact_frac", 0),
        "stuck_s": last.get("stuck_s", float(last.get("contact", 0) or 0) * 0.20),
        "clear": last.get("clear", 0),
        "path": last.get("path", last.get("mean_path", 0)),
        "hits": last.get("hits", 0),
        "liveh": last.get("liveh", 0),
        "ent": last.get("ent", 0),
        "stg": last.get("stg", 0),
        "loss": last.get("loss", 0),
        "ret": last.get("ret", 0),
        "ceil": last.get("ceil") or 0,
        "t_sim": meta.get("t_sim", 0),
        "mtime": meta.get("mtime", 0),
        "play_s": play_s,
        "pack_max_s": pack_max_s,
        "how": meta.get("how", ""),
        "ep_cover": meta.get("rewards", meta.get("cover", 0)),
        "ep_rewards": meta.get("rewards", meta.get("cover", 0)),
        "ep_hits": meta.get("hits", 0),
        "ep_n_free": meta.get("n_free", 0),
        "stats_at": time.time(),
        "series": _series(rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(rec))
    return OUT / "stats.json"


def render_empty_canvas() -> str:
    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""import {{ Callout, H1, Stack, Stat, Text }} from "cursor/canvas";

export default function RlNavLive() {{
  return (
    <Stack gap={{24}}>
      <H1>rl_nav training</H1>
      <Text tone="secondary">
        From-scratch run. Waiting on the first PPO update. Updated {updated}.
      </Text>
      <Stat value="0.0M" label="policy steps" />
      <Callout tone="info">
        Checkpoint, metrics.jsonl, and train.log were parked. Charts refill
        once the new run logs its first update.
      </Callout>
    </Stack>
  );
}}
"""


def write_canvas(rows: list[dict] | None = None) -> Path:
    if rows is None:
        rows = merge_rows(parse_log(LOG), parse_jsonl(JSONL))
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CANVAS.write_text(render_empty_canvas() if not rows else render_canvas(rows))
    return CANVAS


def _clip_mtime() -> float:
    newest = 0.0
    for name in ("last.mp4", "last.gif", "last.png"):
        p = OUT / name
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
    return newest


def _clip_meta() -> tuple[int, int, str]:
    for name in ("last.mp4", "last.gif", "last.png"):
        p = OUT / name
        if not p.is_file():
            continue
        m = re.search(r"clip_s(\d+)_st(\d+)", p.resolve().name)
        if m:
            return int(m.group(1)), int(m.group(2)), p.resolve().name
    return 0, 0, "none"


def _scale_nn(bgr, scale=0.5):
    import cv2
    h, w = bgr.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)


def _frames_from_mp4(path: Path, scale=0.5, stride=2):
    import cv2
    cap = cv2.VideoCapture(str(path.resolve()))
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(_scale_nn(f, scale))
        i += 1
    cap.release()
    return frames


def _frames_from_gif(path: Path, scale=0.5, stride=2):
    import cv2
    import numpy as np
    from PIL import Image
    im = Image.open(path.resolve())
    frames = []
    i = 0
    try:
        while True:
            if i % stride == 0:
                rgb = np.array(im.convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frames.append(_scale_nn(bgr, scale))
            im.seek(im.tell() + 1)
            i += 1
    except EOFError:
        pass
    return frames


def _gif_b64(frames) -> str:
    import cv2
    import numpy as np
    from PIL import Image
    # Force occupancy + SELF + throttle into the shared palette so the
    # hull stays bag-blue instead of mixing to grey/teal under 64 colors.
    anchors = np.array(
        ((18, 18, 22), (40, 170, 90), (210, 50, 40), (40, 90, 220), (255, 210, 0)),
        dtype=np.uint8,
    )
    rgb0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB).copy()
    rgb0[0, :5] = anchors
    pal = Image.fromarray(rgb0).quantize(colors=256, dither=Image.Dither.NONE)
    imgs = []
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if i == 0:
            rgb = rgb.copy()
            rgb[0, :5] = anchors
        imgs.append(Image.fromarray(rgb).quantize(palette=pal, dither=Image.Dither.NONE))
    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=180,
        loop=0,
        optimize=False,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_clip_canvas(b64: str, step: int, stg: int, n: int, src: str) -> str:
    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    step_lab = f"{step / 1e6:.1f}M" if step >= 1_000_000 else str(step)
    return f"""import {{
  Callout,
  Grid,
  H1,
  Stack,
  Stat,
  Text,
  useHostTheme,
}} from "cursor/canvas";

const GIF = "data:image/gif;base64,{b64}";

export default function RlNavClip() {{
  const theme = useHostTheme();
  return (
    <Stack gap={{20}}>
      <Stack gap={{8}}>
        <H1>rl_nav env0 clip</H1>
        <Text tone="secondary">
          60 frames × stride 5 (~60 s of 5 Hz), env 0. Canvas embeds the preview
          GIF (half-res, already paletted). Rewrites when a new clip lands.
          Updated {updated}.
        </Text>
      </Stack>

      <Grid columns={{3}} gap={{16}}>
        <Stat value="{step_lab}" label="clip at policy step" />
        <Stat value="{stg}" label="curriculum stage" />
        <Stat value="{n}" label="frames" />
      </Grid>

      <Callout tone="info">
        House + robot + FOV. Source {src}. Plays looping in this canvas.
      </Callout>

      <img
        src={{GIF}}
        alt="rl_nav env0 clip"
        style={{{{
          width: "100%",
          maxWidth: 520,
          background: theme.fill.tertiary,
          border: "1px solid " + theme.stroke.secondary,
        }}}}
      />
    </Stack>
  );
}}
"""


def write_clip_canvas(force: bool = False) -> Path | None:
    global _CLIP_STAMP
    gif = OUT / "last.gif"
    if not gif.is_file():
        return None
    stamp = gif.stat().st_mtime
    if not force and stamp <= _CLIP_STAMP and CLIP_CANVAS.is_file():
        return CLIP_CANVAS
    # Embed the preview GIF as-is. Do not decode/re-quantize — that was the
    # 1 MB canvas rewrite and the extra minute of CPU.
    b64 = base64.b64encode(gif.read_bytes()).decode("ascii")
    step, stg, src = _clip_meta()
    CLIP_CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CLIP_CANVAS.write_text(render_clip_canvas(b64, step, stg, 60, src))
    _CLIP_STAMP = stamp
    return CLIP_CANVAS


def loop(period_s: float = 30.0):
    while True:
        try:
            write_stats()
        except Exception as e:
            print("stats skip", e, flush=True)
        try:
            p = write_canvas()
            print("chart wrote", p, "rows", len(merge_rows(parse_log(LOG), parse_jsonl(JSONL))), flush=True)
        except Exception as e:
            print("chart skip", e, flush=True)
        try:
            c = write_clip_canvas()
            if c is not None:
                print("clip canvas wrote", c, "bytes", c.stat().st_size, flush=True)
        except Exception as e:
            print("clip canvas skip", e, flush=True)
        time.sleep(period_s)


if __name__ == "__main__":
    write_canvas()
    write_clip_canvas(force=True)
    loop(15.0)
