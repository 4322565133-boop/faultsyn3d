#!/bin/bash
# Archived (hard) dataset, archived 880/110/110 split.  Three ~40-50 M models, old recipe trained to convergence:
# warmup 10 -> hold lr 1e-4, halve after 10 epochs without val-loss improvement, stop after 30 (cap 1000 epochs).
#   1) UNet-L+MaxViT attention scaled to 39.0 M (unet_lg_xl)   2) MaxViT-tiny all3 40.2 M   3) UNet base96 50.4 M
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --legacy --sched plateau --epochs 1000 --patience 30 "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_$1 > logs/eval_legacy_$1.log 2>&1; }
run 29791 --variant unet_lg_xl --tag legacy_unet_lg_xl --per-rank 1 --accum 2 > logs/train_legacy_unet_lg_xl.log 2>&1; ev legacy_unet_lg_xl
run 29792 --variant baseline   --tag legacy_maxvit_all3 --per-rank 2           > logs/train_legacy_maxvit_all3.log 2>&1; ev legacy_maxvit_all3
run 29793 --variant unet_b96   --tag legacy_unet_b96    --per-rank 1 --accum 2 > logs/train_legacy_unet_b96.log 2>&1;   ev legacy_unet_b96
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_legacy_unet_lg_xl runs/maxvit3d_legacy_maxvit_all3 runs/maxvit3d_legacy_unet_b96 > logs/eval_legacy_40m.log 2>&1
