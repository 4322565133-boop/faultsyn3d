#!/bin/bash
# Prepare only: wait for the Thebe download, cut train/val cubes, then test cubes.  No training is started.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
need() { for k in "$@"; do grep -q "OK seis/seis$k.npz" logs/thebe_download.log && grep -q "OK fault/fault$k.npz" logs/thebe_download.log || return 1; done; }
until need train1 train2 train3 train4 train5 train6 train7 train8 train9 val1 val2; do sleep 60; done
echo "train/val files complete $(date)"
$PY synth/thebe_cubes.py --out data/thebe_cubes --n-train 2400 --n-val 240 --splits train val > logs/thebe_cubes_trainval.log 2>&1 || { echo "cube cutting (train/val) failed"; exit 1; }
echo "train/val cubes done $(date)"
until need test1 test2 test3 test4 test5 test6 test7; do sleep 60; done
$PY synth/thebe_cubes.py --out data/thebe_cubes --splits test > logs/thebe_cubes_test.log 2>&1 || { echo "cube cutting (test) failed"; exit 1; }
echo "THEBE READY $(date)"
