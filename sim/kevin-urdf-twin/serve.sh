#!/usr/bin/env bash
# Serve the Kevin URDF twin for browser / phone viewing.
# Usage: ./serve.sh [port]
set -euo pipefail
PORT="${1:-8080}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "Serving Kevin twin at http://0.0.0.0:${PORT}/viewer/"
echo "  URDF:  ${ROOT}/urdf/kevin.urdf"
echo "  Open on phone via this host's LAN IP."
exec python3 -m http.server "$PORT" --bind 0.0.0.0
