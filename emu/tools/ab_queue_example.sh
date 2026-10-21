#!/bin/bash
# fixed safety suite, then clean chi2 (motion-model cov) vs old gate, room/maze/hall seeds 0-2; ends with live.sh
pkill -f "m emu.world"; pkill -f "m nav.run"; pkill -f "viewer.py 8095"; sleep 3
cd "$(dirname "$0")/../.."
O=~/.kevin/emu/safety_fix2; KEVIN_AUTOCAL=0 ~/.venv-kevinsim/bin/python -m emu.safety_eval --env room --out $O > $O.log 2>&1
r(){ tag=$1; s=$2; shift 2; env "$@" KEVIN_EMU_NOISE=realistic ~/.venv-kevinsim/bin/python -m emu.eval --envs room maze hall --seed $s --out ~/.kevin/emu/abc_$tag >> ~/.kevin/emu/abc_$tag.log 2>&1; }
for s in 0 1 2; do r chi2m $s KEVIN_SLAM2_TEST=chi2; r gate $s KEVIN_SLAM2_TEST=gate; done
touch ~/.kevin/emu/abclean_DONE; emu/viewer/live.sh
