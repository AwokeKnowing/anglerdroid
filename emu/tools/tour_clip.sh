#!/bin/bash
# usage: tour_clip.sh OUT env1 env2 ...   -> OUT/<env>_s0 + OUT.log + clips/<env>/frames
OUT=$1; shift; cd "$(dirname "$0")/../.."
for e in "$@"; do
  ~/.venv-kevinsim/bin/python -m emu.eval --envs $e --out $OUT >> $OUT.log 2>&1 &
  EP=$!; rm -rf ~/.kevin/emu/clips/$e$CLIPTAG; mkdir -p ~/.kevin/emu/clips/$e$CLIPTAG
  for k in $(seq 60); do sleep 2; ~/.venv-kevinrobot/bin/python -c "import urllib.request;urllib.request.urlopen('http://localhost:8090/api/state',timeout=1)" 2>/dev/null && break; done
  KEVIN_HAL=sim scripts/kevin-hal ~/.venv-kevinrobot/bin/python emu/tools/clip_rec.py ~/.kevin/emu/clips/$e$CLIPTAG 8090 4 > ~/.kevin/emu/clips/$e$CLIPTAG.rec.log 2>&1 &
  RP=$!; wait $EP; touch ~/.kevin/emu/clips/$e$CLIPTAG/STOP; sleep 3; kill $RP 2>/dev/null
done
