#!/bin/bash
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29602 \
    train/train_old_recipe_ddp.py --variant dyn --tag dyn_ddp > logs/train_dyn_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_dyn_ddp > logs/eval_dyn.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/analyze_runs.py --gpu 0 --out logs/analyze_dyn.txt --runs runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_dyn_ddp >> logs/eval_dyn.log 2>&1
