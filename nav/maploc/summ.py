import json, sys
r = json.load(open(sys.argv[1] + "/summary.json"))
for n, x in r.items():
    print(n, "path %.1fm %.0fs scale %.4f+-%.4f" % (x["path_m"], x["dur_s"], x.get("map_scale_final", 1), x.get("map_scale_sd", 0)), " | ".join("%s pos med/p90/final %.3f/%.3f/%.3f yaw p90 %.2f" % (k, x[k]["pos_med"], x[k]["pos_p90"], x[k]["pos_final"], x[k]["yaw_p90"]) for k in ("odom", "slam2", "maploc")), "| acc %d rej %.2f %s | %.0f ms c=%.4f" % (x["accepted"], x["reject_rate"], x["why"], x["ms_med"], x.get("info_scale", 1)))
