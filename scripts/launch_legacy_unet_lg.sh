#!/bin/bash
# Archived (hard) dataset faults6_stratified_1100, same 880/110/110 split as the archived MaxViT-40M run (test 0.611).
#   1) UNet-L + MaxViT-style attention (unet_lg, 11.9M)   2) UNet-L (9.6M) capacity control   3) legacy test table
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --legacy "${@:2}"; }
run 29731 --variant unet_lg --tag legacy_unet_lg --per-rank 2 > logs/train_legacy_unet_lg.log 2>&1
run 29732 --variant unet_l  --tag legacy_unet_l  --per-rank 2 > logs/train_legacy_unet_l.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_legacy_unet_lg runs/maxvit3d_legacy_unet_l runs/maxvit3d_dbvit_legacy_ddp > logs/eval_legacy_unet_lg.log 2>&1
