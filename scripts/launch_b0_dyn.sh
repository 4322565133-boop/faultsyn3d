#!/bin/bash
# New protocol (DDP x4, fp32 seg loss, BN-synced val): B0 then B0 + dynamic upsample, from scratch, 200 ep each.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29601 \
    train/train_old_recipe_ddp.py --variant baseline --tag b0_ddp > logs/train_b0_ddp.log 2>&1
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29602 \
    train/train_old_recipe_ddp.py --variant dyn --tag dyn_ddp > logs/train_dyn_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_b0_ddp runs/maxvit3d_dyn_ddp > logs/eval_b0_dyn.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/analyze_runs.py --gpu 0 --out logs/analyze_b0_dyn.txt --runs runs/maxvit3d_b0_ddp runs/maxvit3d_dyn_ddp >> logs/eval_b0_dyn.log 2>&1
