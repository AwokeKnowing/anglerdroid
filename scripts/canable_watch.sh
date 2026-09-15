#!/bin/bash
# Soft-reset the USB CANable branch if 1d50:606f is gone.
# Power-cycles VIA hub 1-2.3 port 3 only (Terminus: CANable + PeakDo + ReSpeaker).
# Does NOT touch 1-2.3 port 4 (WiFi + RealSense).
# Cron (root, every minute): /home/jetbot/anglerdroid/scripts/canable_watch.sh
set -eu

VID="1d50"
PID="606f"
# USB2 VIA hub whose port 3 is the Terminus that holds the CANable.
HUB="1-2.3"
PORT="3"
STAMP="/home/jetbot/.kevin/logs/canable_watch.stamp"
LOG="/home/jetbot/.kevin/logs/canable_watch.log"
LOCK="/tmp/canable_watch.lock"
MIN_RESET_S=180
BOOT_GRACE_S=90

mkdir -p "$(dirname "$STAMP")" "$(dirname "$LOG")"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG"
}

if [ "$(awk '{print int($1)}' /proc/uptime)" -lt "$BOOT_GRACE_S" ]; then
  exit 0
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

find_canable() {
  local d
  for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] && [ -f "$d/idProduct" ] || continue
    if [ "$(cat "$d/idVendor")" = "$VID" ] && [ "$(cat "$d/idProduct")" = "$PID" ]; then
      printf '%s\n' "$d"
      return 0
    fi
  done
  return 1
}

gs_usb_up() {
  local n
  for n in /sys/class/net/can*; do
    [ -e "$n" ] || continue
    if grep -q '^DRIVER=gs_usb' "$n/device/uevent" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

if find_canable >/dev/null && gs_usb_up; then
  exit 0
fi

now=$(date +%s)
if [ -f "$STAMP" ]; then
  last=$(cat "$STAMP" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt "$MIN_RESET_S" ]; then
    log "canable missing; skip reset (debounce ${MIN_RESET_S}s)"
    exit 0
  fi
fi

echo "$now" >"$STAMP"
log "canable ${VID}:${PID} missing or no gs_usb; uhubctl -l ${HUB} -p ${PORT} cycle"
if ! command -v uhubctl >/dev/null; then
  log "uhubctl not installed"
  exit 1
fi
timeout 15 uhubctl -l "$HUB" -p "$PORT" -a cycle -d 1 >>"$LOG" 2>&1 || {
  log "uhubctl cycle failed rc=$?"
  exit 1
}
sleep 2
if find_canable >/dev/null; then
  log "canable back after hub cycle"
else
  log "canable still missing after hub cycle"
fi
