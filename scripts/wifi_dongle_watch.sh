#!/bin/bash
# If Kevin has no LAN on the Realtek USB dongle, unplug/replug that device only.
# Sibling hub ports (RealSense / cameras) are left alone.
# Cron (root, every minute): /home/jetbot/anglerdroid/scripts/wifi_dongle_watch.sh
set -eu

VID="0bda"
PID="8812"
LAN="192.168.50"
GW="192.168.50.1"
STAMP="/home/jetbot/.kevin/logs/wifi_watch.stamp"
LOG="/home/jetbot/.kevin/logs/wifi_watch.log"
LOCK="/tmp/wifi_dongle_watch.lock"
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

lan_ok() {
  local iface
  for iface in /sys/class/net/wl*; do
    [ -e "$iface" ] || continue
    iface=$(basename "$iface")
    ip -4 -br addr show "$iface" 2>/dev/null | grep -q "$LAN" || continue
    if ping -c 1 -W 2 -I "$iface" "$GW" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

find_dongle() {
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

if lan_ok; then
  exit 0
fi

now=$(date +%s)
if [ -f "$STAMP" ]; then
  last=$(cat "$STAMP" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt "$MIN_RESET_S" ]; then
    log "lan down; skip reset (debounce ${MIN_RESET_S}s)"
    exit 0
  fi
fi

dev=$(find_dongle || true)
if [ -z "${dev:-}" ]; then
  log "lan down; dongle ${VID}:${PID} not in sysfs"
  exit 0
fi

name=$(basename "$dev")
log "lan down; reset usb $name ($dev)"
echo "$now" >"$STAMP"
# authorized 0/1 is the software unplug/replug. Do NOT reset parent hub
# 1-2.3.4 — cameras share that hub.
if [ -w "$dev/authorized" ]; then
  echo 0 >"$dev/authorized"
  sleep 2
  echo 1 >"$dev/authorized"
  log "usb $name authorized cycle done"
else
  log "cannot write $dev/authorized (need root)"
  exit 1
fi
