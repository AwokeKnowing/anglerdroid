#!/bin/bash
# usage: stopB.sh <prefix> ; uses KEVIN_REPO from env
cd ~/m1_backup/dbg; PY=~/.venv-kevinrobot/bin/python
$PY stoptest.py ${1}_head 0.30 0.45 0.60
for dl in 0 0.2 -0.2; do W=-0.4 DL=$dl $PY stoptest.py ${1}_arcR04_dl$dl 0.60; done
