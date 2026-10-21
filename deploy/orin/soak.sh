#!/bin/bash
# stationary soak on the Orin (no-motor stack): coreage per 60 s window, tegrastats, command cycle (stub wheels only)
# Run from ~/anglerdroid or set KEVIN_REPO. Copies live in ~/kevin_rollback/ on the Orin.
# usage: soak.sh <tag> <minutes> [load=1]
T=$1; M=${2:-10}; LOAD=${3:-1}; D=~/kevin_bench/soak_$T; mkdir -p $D; cd "${KEVIN_REPO:-$HOME/anglerdroid}"
J0=$(date '+%Y-%m-%d %H:%M:%S')
timeout $((M*60+5)) tegrastats --interval 5000 > $D/tegrastats.txt 2>&1 &
# the command cycle SENDS DRIVE COMMANDS: only against the no-motor stub stack, never a real nav
if [ "$LOAD" = 1 ] && ! pgrep -f "python hal/nomotor/run_nomotor.py" > /dev/null; then echo "soak: no hal/nomotor stack running; refusing the drive-command cycle (use load=0)"; exit 1; fi
if [ "$LOAD" = 1 ] && pgrep -f "python -m nav.run" > /dev/null; then echo "soak: a real nav.run is running; refusing the drive-command cycle"; exit 1; fi
if [ "$LOAD" = 1 ]; then ( end=$((SECONDS+M*60)); while [ $SECONDS -lt $end ]; do
  for c in "drive 0 0.8 5" "stop" "drive 0.15 0 3" "stop" "forward 1" "stop" "drive 0 -0.8 5" "stop"; do
    curl -s -m 3 localhost:8080/api/command -H 'Content-Type: application/json' -d "{\"text\":\"$c\"}" > /dev/null 2>&1 || python3 -c "import urllib.request as u,json,sys;u.urlopen(u.Request('http://localhost:8080/api/command',data=json.dumps({'text':sys.argv[1]}).encode(),headers={'Content-Type':'application/json'}),timeout=3).read()" "$c" > /dev/null 2>&1
    sleep 6; done; done ) & fi
for i in $(seq $M); do ~/.venv-kevinnav/bin/python -m emu.tools.scbench.coreage 60 --json $D/w$i.json > /dev/null; done
python3 - $D <<'PY'
import json, glob, sys
W = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/w*.json"), key=lambda f: int(f.split("/w")[-1][:-5]))]
codes = {}
for w in W:
    for k, v in w["codes"].items(): codes[k] = codes.get(k, 0) + v
print("windows %d  hz %.2f-%.2f  age1 p50 %.1f-%.1f p99 %.1f-%.1f max %.1f | age2 p50 %.1f-%.1f p99 max %.1f max %.1f | late>50ms %d | RS1 frames dropped %d | over100 %d | codes %s" % (
  len(W), min(w["hz"] for w in W), max(w["hz"] for w in W), min(w["age1_ms"]["p50"] for w in W), max(w["age1_ms"]["p50"] for w in W),
  min(w["age1_ms"]["p99"] for w in W), max(w["age1_ms"]["p99"] for w in W), max(w["age1_ms"]["max"] for w in W),
  min(w["age2_ms"]["p50"] for w in W), max(w["age2_ms"]["p50"] for w in W), max(w["age2_ms"]["p99"] for w in W), max(w["age2_ms"]["max"] for w in W),
  sum(w.get("late_50ms", w.get("missed_50ms", 0)) for w in W), sum(w.get("rs1_dropped", -1) for w in W), sum(w["over_100ms"] for w in W), codes))
PY
awk '{for(i=1;i<=NF;i++) if($i ~ /^tj@/){split($i,a,"@"); sub("C","",a[2]); split(a[2],b,"/"); if(b[1]>m) m=b[1]}} END{print "tegrastats max tj", m, "C"}' $D/tegrastats.txt
grep -o "CPU \[[^]]*\]" $D/tegrastats.txt | tail -1
journalctl -k --since "$J0" --no-pager | grep -iE "usb|xhci|uvc|thrott|error" | grep -viE "Unknown video format|Found UVC|hid-generic" | tail -5; echo "kernel usb/err lines: $(journalctl -k --since "$J0" --no-pager | grep -iE 'usb|xhci|uvc|thrott|error' | grep -viE 'Unknown video format|Found UVC|hid-generic' | wc -l)"
grep -ciE "stuck|stale|error|Traceback" ~/kevin_bench/nav_latest.log
