#!/bin/bash
# Keep offline jobs off the CPU while a sim recording runs (the sim colour lane, physics and nav share cpus 0,1,4,5; replays
# there cut colour to 19-22 Hz on 2026-10-19).  Usage:  record_guard.sh <command...>
# SIGSTOPs every matching offline job every 2 s while <command> runs (so jobs started later are caught too), SIGCONTs them
# all when it ends (also on Ctrl-C / kill).  Writes /tmp/kevin_recording.lock (pid) so job launchers can wait on it.
PAT=${KEVIN_OFFLINE_PAT:-"emu\.tools\.slamlog\.|emu/tools/slamlog/|lslam_replay|kslam_replay|tilt_eval|spin_refiner|spin_lodo|/tmp/cc/mkcache|/tmp/cc/.*pipe\.sh"}
LOCK=/tmp/kevin_recording.lock; STOPPED=/tmp/kevin_recording.stopped; : > $STOPPED; echo $$ > $LOCK
stop_all() { for p in $(pgrep -f "$PAT"); do [ "$p" = $$ ] && continue; kill -STOP $p 2>/dev/null && echo $p >> $STOPPED; done; }
cont_all() { for p in $(sort -u $STOPPED); do kill -CONT $p 2>/dev/null; done; rm -f $LOCK $STOPPED; }
trap 'cont_all' EXIT INT TERM
stop_all; ( while [ -f $LOCK ]; do sleep 2; stop_all; done ) & G=$!
"$@"; rc=$?
rm -f $LOCK; kill $G 2>/dev/null; wait $G 2>/dev/null; exit $rc
