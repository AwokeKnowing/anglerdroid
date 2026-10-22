"""Aggregate RESULTS/<seq>/{scores.json, maps.json} over n runs -> markdown table (medians, min..max).
python bench/slam/summarize.py OUT.md RESULTS/tour_s0 RESULTS/tour_s1 ..."""
import json, os, re, sys
import numpy as np
out, dirs = sys.argv[1], sys.argv[2:]; R = {}
for d in dirs:
    S = {r["method"]: r for r in json.load(open(os.path.join(d, "scores.json"))) if r.get("ok")}
    M = json.load(open(os.path.join(d, "maps.json"))) if os.path.exists(os.path.join(d, "maps.json")) else {}
    for k, r in S.items():
        rt = r.get("runtime") or ""; m = re.search(r"\(([\d.]+) fps\)", rt) or re.search(r"([\d.]+) input fps", rt) or re.search(r"\(([\d.]+) fps", rt)
        ms = re.search(r"([\d.]+) ms/(frame|scan)", rt); fps = float(m.group(1)) if m else (1e3 / float(ms.group(1)) if ms else None)
        R.setdefault(k, []).append(dict(ate=r["ate_xy"], fin=r["final_xy"], yaw=r["yaw_max"], rpe=r["rpe1m_med"], al=r["ate_aligned"], cov=r.get("coverage"),
                                        iou=M.get(k, {}).get("occ_iou"), f1=M.get(k, {}).get("occ_f1_t"), fiou=M.get(k, {}).get("free_iou"), fps=fps))
    if "_gt_pose_map" in M: R.setdefault("_gt_pose_map (mapper upper bound)", []).append(dict(iou=M["_gt_pose_map"]["occ_iou"], f1=M["_gt_pose_map"]["occ_f1_t"], fiou=M["_gt_pose_map"]["free_iou"]))
def med(v, k, f="%.3f", rng=True):
    x = [r.get(k) for r in v if r.get(k) is not None]
    if not x: return "-"
    s = f % np.median(x); return s + (" (%s..%s)" % (f % min(x), f % max(x)) if rng and len(x) > 1 else "")
rows = sorted(R.items(), key=lambda kv: np.median([r["ate"] for r in kv[1]]) if "ate" in kv[1][0] else -1)
L = ["| method | n | ATE xy m, median (min..max) | final xy m | max yaw err deg | RPE/1m m | coverage | map occ IoU | map occ F1 (4 cm tol) | free IoU | fps* |", "|" + "---|" * 11]
for k, v in rows:
    L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (k, len(v), med(v, "ate"), med(v, "fin", rng=False), med(v, "yaw", "%.1f", False), med(v, "rpe", rng=False),
             med(v, "cov", "%.2f", False), med(v, "iou"), med(v, "f1", rng=False), med(v, "fiou", rng=False), med(v, "fps", "%.1f", False)))
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
