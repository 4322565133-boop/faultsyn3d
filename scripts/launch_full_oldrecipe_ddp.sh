#!/bin/bash
# Full upgraded model (D2S + profile + trace), archived recipe on 4 GPUs, from scratch, 200 epochs.
# Baseline for comparison: runs/maxvit3d_v2_oldrecipe_all3 (test IoU 0.833).
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29581 \
    train/train_old_recipe_ddp.py --variant full --tag full_oldrecipe_ddp > logs/train_full_oldrecipe_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_full_oldrecipe_ddp \
    > logs/eval_full_oldrecipe_ddp.log 2>&1
