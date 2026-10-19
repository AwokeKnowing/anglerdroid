#!/bin/bash
# Sim CPU layout on i777 (i7-7700, 4 cores / 8 threads; HT siblings n and n+4), user-level only (no root):
#   safety path: physical core 3 (cpus 3+7) = nav safety core on 3 (KEVIN_SC_CPU) + nav RS2 camera process on its sibling 7
#                (light: one copy per frame), so the core's ~31 ms/frame of numpy is not shared with a busy hyperthread
#                physical core 2 (cpus 2+6) = emu.depthcam RS1 lane on 6, RS2 lane on 2 (KEVIN_EMU_DEPTH_CPUS)
#   cpus 0,1,4,5 = everything else: emu.world physics + colour/viewer lane, nav brain, viewer, LightNav, ollama
#   LightNav / ollama / viewer: nice +10 / +10 / +10 (lower priority; raising priority needs root, not done)
# usage: source simpin.sh            -> exports the env for launches (restart_world.sh / restart_nav.sh read it)
#        simpin.sh apply             -> also re-pins already running processes (all threads)
export SIM_OTHER_CPUS=${SIM_OTHER_CPUS:-0,1,4,5} KEVIN_EMU_DEPTH_CPUS=${KEVIN_EMU_DEPTH_CPUS:-6,2} KEVIN_SC_CPU=${KEVIN_SC_CPU:-3} KEVIN_RS2CAM_CPUS=${KEVIN_RS2CAM_CPUS:-7}
export WORLD_PREFIX="taskset -c $SIM_OTHER_CPUS" NAV_PREFIX="taskset -c $SIM_OTHER_CPUS"
pin() { for p in "$@"; do taskset -a -p -c "$SIM_OTHER_CPUS" "$p" > /dev/null && echo "pinned $p ($(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-60)) -> $SIM_OTHER_CPUS"; done; }
nicer() { local n=$1; shift; for p in "$@"; do for t in /proc/$p/task/*; do renice -n "$n" -p "${t##*/}" > /dev/null 2>&1; done; echo "nice +$n $p"; done; }
if [ "$1" = apply ]; then
  pin $(pgrep -f "python -m emu.world") $(pgrep -f "nav.run --port 8090") $(pgrep -f "viewer/viewer.py 8095") $(pgrep -f lightnav-serve) $(pgrep -f "ollama serve")
  : # depthcam pins its own lanes (KEVIN_EMU_DEPTH_CPUS) at start: restart the world to change it
  nicer 10 $(pgrep -f lightnav-serve) $(pgrep -f "ollama serve"); nicer 10 $(pgrep -f "viewer/viewer.py 8095")
  # desktop apps (Chrome, Grok Bot app) were measured on cpus 2/6/7 at 30-50 % of a cpu each, 7 being the safety core's
  # hyperthread sibling -> core frame-age spikes >100 ms (2026-10-19 08:40).  Keep them on the 'everything else' cpus too
  # (children inherit).  SIMPIN_DESKTOP=0 to skip; undo: taskset -a -p -c 0-7 <pid>.
  if [ "${SIMPIN_DESKTOP:-1}" = 1 ]; then
    # NOT the Grok Bot app: agent shells are its children and would inherit the nice (and a user cannot lower it back)
    D="$(pgrep -u "$USER" -x chrome)"
    for p in $D $(pgrep -u "$USER" -x cloudflared); do taskset -a -p -c "$SIM_OTHER_CPUS" "$p" > /dev/null 2>&1; done
    # on 0,1,4,5 (= 2 physical cores) they starved emu.world's 1 kHz loop: camera poses written 30-96 ms late (CAMLOG):
    # lowest priority for the desktop apps (user renice, no root)
    nicer 10 $D > /dev/null; echo "desktop apps pinned -> $SIM_OTHER_CPUS, nice +10"
  fi
fi
