#!/bin/bash
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
while pgrep -f "train/train.py --model maxvit3d --gpu 1" >/dev/null; do sleep 60; done
echo "=== GPU1 free $(date): resnet old-recipe ===" >> logs/queue2.log
nohup $PY train/train_old_recipe.py --model resnet3d --gpu 1 > logs/train_oldrecipe_resnet3d.log 2>&1 &
while pgrep -f "train/train.py --model maxvit3d --gpu 2" >/dev/null; do sleep 60; done
echo "=== GPU2 free $(date): analysis + learning curve ===" >> logs/queue2.log
$PY train/analyze_runs.py --gpu 2 --out logs/analyze_maxvit_fixes.txt \
    --runs runs/maxvit3d_v2 runs/maxvit3d_v2_dp02 runs/maxvit3d_v2_frs runs/maxvit3d_v2_p4 runs/maxvit3d_v2_all3 runs/unet3d_v2 > logs/analyze_maxvit_fixes.log 2>&1
$PY train/train.py --model unet3d   --gpu 2 --tag v2_n200 --limit 200 > logs/train_v2_unet_n200.log   2>&1
$PY train/train.py --model maxvit3d --gpu 2 --tag v2_n200 --limit 200 > logs/train_v2_maxvit_n200.log 2>&1
$PY train/train.py --model unet3d   --gpu 2 --tag v2_n400 --limit 400 > logs/train_v2_unet_n400.log   2>&1
$PY train/train.py --model maxvit3d --gpu 2 --tag v2_n400 --limit 400 > logs/train_v2_maxvit_n400.log 2>&1
echo "=== all queued work done $(date) ===" >> logs/queue2.log
