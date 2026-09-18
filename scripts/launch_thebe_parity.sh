#!/bin/bash
# Thebe field data, 128^3 cubes (train 2400 / val 240, crossline split of the data descriptor), OLD RECIPE DEFAULTS
# (dice+focal, AdamW wd .01, warmup 10 -> 1e-4 cosine -> 1e-7 over 200, global batch 8, val-loss selection, patience 20).
# Two compute-parity tiers; after each run: cube-grid test (evaluate.py --thebe) + full-volume sliding-window test (infer_thebe.py).
#   0.40 TF tier: MaxViT-pico all3 9.8M | UNet base22 2.65M | UNet+MaxViT-attn base22 3.30M
#   0.77 TF tier: MaxViT-tiny all3 40.2M | UNet base30 4.9M | UNet+MaxViT-attn base30 6.1M
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --thebe "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --thebe --runs runs/maxvit3d_$1 > logs/eval_thebe_$1.log 2>&1
       CUDA_VISIBLE_DEVICES=0 $PY train/infer_thebe.py --gpu 0 --runs runs/maxvit3d_$1 > logs/infer_thebe_$1.log 2>&1; }
run 29831 --variant maxvit_pico --tag thebe_maxvit_pico --per-rank 2 > logs/train_thebe_maxvit_pico.log 2>&1; ev thebe_maxvit_pico
run 29832 --variant unet_b22    --tag thebe_unet_b22    --per-rank 2 > logs/train_thebe_unet_b22.log 2>&1;    ev thebe_unet_b22
run 29833 --variant unet_lg_b22 --tag thebe_unet_lg_b22 --per-rank 2 > logs/train_thebe_unet_lg_b22.log 2>&1; ev thebe_unet_lg_b22
run 29834 --variant baseline    --tag thebe_maxvit_tiny --per-rank 2 > logs/train_thebe_maxvit_tiny.log 2>&1; ev thebe_maxvit_tiny
run 29835 --variant unet_b30    --tag thebe_unet_b30    --per-rank 2 > logs/train_thebe_unet_b30.log 2>&1;    ev thebe_unet_b30
run 29836 --variant unet_lg_b30 --tag thebe_unet_lg_b30 --per-rank 2 > logs/train_thebe_unet_lg_b30.log 2>&1; ev thebe_unet_lg_b30
CUDA_VISIBLE_DEVICES=0 $PY train/infer_thebe.py --gpu 0 --no-save --runs runs/maxvit3d_thebe_maxvit_pico runs/maxvit3d_thebe_unet_b22 runs/maxvit3d_thebe_unet_lg_b22 runs/maxvit3d_thebe_maxvit_tiny runs/maxvit3d_thebe_unet_b30 runs/maxvit3d_thebe_unet_lg_b30 > logs/infer_thebe_all.log 2>&1
