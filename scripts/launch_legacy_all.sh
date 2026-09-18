#!/bin/bash
# Archived (hard) dataset faults6_stratified_1100, archived split 880/110/110, old recipe but trained to convergence
# (warmup 10 -> hold 1e-4, halve on 10-epoch val-loss plateau, stop after 30 stale epochs; no 200-epoch horizon), 4 GPUs.
# Archived references: UNet-S 1.4M test 0.539 | ResNet3D 195M 0.578 | MaxViT-tiny 40M 0.611.
#   1) MaxViT-pico all3  9.8M   2) UNet-L 9.6M   3) UNet-L + MaxViT attention 11.9M   4) MaxViT-nano all3 24.8M
#   5) legacy test table for all of them + DoubleBlock-ViT (stopped @65)
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --legacy --sched plateau --epochs 1000 --patience 30 "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_$1 > logs/eval_legacy_$1.log 2>&1; }
run 29751 --variant maxvit_pico --tag legacy_maxvit_pico --per-rank 2 > logs/train_legacy_maxvit_pico.log 2>&1; ev legacy_maxvit_pico
run 29752 --variant unet_l      --tag legacy_unet_l      --per-rank 2 > logs/train_legacy_unet_l.log 2>&1;      ev legacy_unet_l
run 29753 --variant unet_lg     --tag legacy_unet_lg     --per-rank 2 > logs/train_legacy_unet_lg.log 2>&1;     ev legacy_unet_lg
run 29754 --variant maxvit_nano --tag legacy_maxvit_nano --per-rank 2 > logs/train_legacy_maxvit_nano.log 2>&1; ev legacy_maxvit_nano
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_legacy_maxvit_pico runs/maxvit3d_legacy_unet_l runs/maxvit3d_legacy_unet_lg runs/maxvit3d_legacy_maxvit_nano runs/maxvit3d_dbvit_legacy_ddp > logs/eval_legacy_all.log 2>&1
