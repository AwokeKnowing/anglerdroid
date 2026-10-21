#!/bin/bash
# Run offline CPU work (replays, evals, sensorlog_pack) ONLY while the sim is stopped:  emu/tools/offline_gate.sh <cmd...>
# Why: even pinned to cpus 0,1,4,5 at SCHED_IDLE + nice 19, the 2026-10-20 tilt/lslam jobs raised the safety core's frame
# age (paused p99 76 / max 82 ms -> running p99 100-119 / max 104-135): idle-class CPU still shares hyperthreads, L3 and
# memory bandwidth with the sim, and nothing in the scheduler can isolate that.  So the job is SIGSTOPped whenever the
# sim (emu.world) runs or a recording holds /tmp/kevin_recording.lock, and SIGCONTed when both are gone (checked every 2 s).
# GPU-only work does not need this gate.
desc() { echo $1; for c in $(pgrep -P $1); do desc $c; done; }
"$@" & P=$!; S=run
trap 'for p in $(desc $P); do kill -TERM $p; kill -CONT $p; done 2>/dev/null; exit 143' TERM INT HUP   # never leave it frozen
while kill -0 $P 2>/dev/null; do
  if pgrep -f "emu[.]world|emu/world[.]py" >/dev/null || [ -e /tmp/kevin_recording.lock ]; then want=stop; else want=run; fi
  if [ $want != $S ]; then
    for p in $(desc $P); do kill -$([ $want = stop ] && echo STOP || echo CONT) $p 2>/dev/null; done
    S=$want; echo "offline_gate: $want ($(date +%T))" >&2
  fi
  sleep 2
done; wait $P
