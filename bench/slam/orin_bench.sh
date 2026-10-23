#!/bin/bash
# Run the slam2 replay timing suite on this box (Orin or i777).  Never touches motors / nav / pilot: pure offline replay.
# usage: PY=python3 bench/slam/orin_bench.sh JOBS.pkl.gz OUTDIR
set -u; JOBS=$1; OUT=$2; PY=${PY:-python3}; mkdir -p "$OUT"; cd "$(dirname "$0")/../.."
export KEVIN_MUTE=1
(nvpmodel -q 2>&1; cat /etc/nv_tegra_release 2>/dev/null; uname -a; nproc; free -m) > "$OUT/sys.txt" 2>&1
TS=""; if command -v tegrastats >/dev/null; then tegrastats --interval 500 --logfile "$OUT/tegrastats.log" & TS=$!; fi
run() { tag=$1; shift; date +%s.%N > "$OUT/$tag.t0"; $PY bench/slam/orin_bench.py "$JOBS" --out "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1; date +%s.%N > "$OUT/$tag.t1"; tail -c 300 "$OUT/$tag.log"; echo; }
run life1 --life 1
run life0 --life 0
run life1_core --life 1 --core-load
run life1_rt_core --life 1 --core-load --realtime
[ -n "$TS" ] && kill $TS 2>/dev/null; command -v tegrastats >/dev/null && tegrastats --stop 2>/dev/null
echo done
