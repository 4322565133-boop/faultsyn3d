#!/bin/bash
# Compute-parity benchmark on the archived (hard) dataset, archived 880/110/110 split, old recipe trained to
# convergence (plateau schedule).  Anchor = MaxViT-tiny all3, 0.77 TFLOPs per 128^3 volume.
#   A (same compute)  : MaxViT-tiny all3 40.2M/0.77TF | UNet base30 4.9M/0.79TF | UNet+MaxViT-attn base30 6.1M/0.81TF | MaxViT-nano 24.8M/0.72TF
#   B (same params)   : MaxViT-pico 9.8M/0.40TF | UNet-L base42 9.6M/1.54TF | UNet+attn base42 11.9M/1.59TF
#   C (UNet with 10x compute): UNet base96 50.4M/8.0TF
# After each run: legacy test (in-distribution) + zero-shot Wu released 20 + zero-shot our Wu-style test 100.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --legacy --sched plateau --epochs 1000 --patience 30 "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_$1 > logs/eval_legacy_$1.log 2>&1
       CUDA_VISIBLE_DEVICES=0 $PY train/eval_cross.py --gpu 0 --sets faultseg3d --out logs/eval_wureal_$1.json --runs runs/maxvit3d_$1 > logs/eval_wureal_$1.log 2>&1
       CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --wu --runs runs/maxvit3d_$1 > logs/eval_wusynth_$1.log 2>&1; }
# A
run 29801 --variant baseline     --tag par_maxvit_tiny --per-rank 2           > logs/train_par_maxvit_tiny.log 2>&1; ev par_maxvit_tiny
run 29802 --variant unet_b30     --tag par_unet_b30    --per-rank 2           > logs/train_par_unet_b30.log 2>&1;    ev par_unet_b30
run 29803 --variant unet_lg_b30  --tag par_unet_lg_b30 --per-rank 2           > logs/train_par_unet_lg_b30.log 2>&1; ev par_unet_lg_b30
# B
run 29804 --variant maxvit_pico  --tag par_maxvit_pico --per-rank 2           > logs/train_par_maxvit_pico.log 2>&1; ev par_maxvit_pico
run 29805 --variant unet_l       --tag par_unet_l      --per-rank 2           > logs/train_par_unet_l.log 2>&1;      ev par_unet_l
run 29806 --variant unet_lg      --tag par_unet_lg     --per-rank 2           > logs/train_par_unet_lg.log 2>&1;     ev par_unet_lg
# A (cont.) and C
run 29807 --variant maxvit_nano  --tag par_maxvit_nano --per-rank 2           > logs/train_par_maxvit_nano.log 2>&1; ev par_maxvit_nano
run 29808 --variant unet_b96     --tag par_unet_b96    --per-rank 1 --accum 2 > logs/train_par_unet_b96.log 2>&1;    ev par_unet_b96
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_par_maxvit_tiny runs/maxvit3d_par_unet_b30 runs/maxvit3d_par_unet_lg_b30 runs/maxvit3d_par_maxvit_pico runs/maxvit3d_par_unet_l runs/maxvit3d_par_unet_lg runs/maxvit3d_par_maxvit_nano runs/maxvit3d_par_unet_b96 > logs/eval_legacy_parity_all.log 2>&1
