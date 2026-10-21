#!/bin/bash
# stop tests (head-on + curved R 1.5 m) at 0.30 / 0.45 / 0.60 m/s for the nav in KEVIN_REPO.  usage: stopcore.sh <prefix>
cd ~/m1_backup/dbg; PY=~/.venv-kevinrobot/bin/python
$PY stoptest.py ${1}_head 0.30 0.45 0.60
for v in 0.30 0.45 0.60; do W=$(python3 -c "print(-$v/1.5)") $PY stoptest.py ${1}_arcR15 $v; done
