#!/bin/bash
# Thebe field data, 128^3 cubes, OLD RECIPE DEFAULTS with the epoch cap at 100 (warmup 10 -> cosine to 100, patience 20).
#   MaxViT-tiny all3 40.2M / 0.77 TF   vs   UNet base30 4.9M / 0.79 TF  (compute-matched UNet in the 3-9 M range)
# After each run: 718-cube test grid + full-volume sliding-window test.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --thebe --epochs 100 "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --thebe --runs runs/maxvit3d_$1 > logs/eval_thebe_$1.log 2>&1
       CUDA_VISIBLE_DEVICES=0 $PY train/infer_thebe.py --gpu 0 --runs runs/maxvit3d_$1 > logs/infer_thebe_$1.log 2>&1; }
run 29841 --variant baseline --tag thebe_maxvit_tiny --per-rank 2 > logs/train_thebe_maxvit_tiny.log 2>&1; ev thebe_maxvit_tiny
run 29842 --variant unet_b30 --tag thebe_unet_b30    --per-rank 2 > logs/train_thebe_unet_b30.log 2>&1;    ev thebe_unet_b30
CUDA_VISIBLE_DEVICES=0 $PY train/infer_thebe.py --gpu 0 --no-save --runs runs/maxvit3d_thebe_maxvit_tiny runs/maxvit3d_thebe_unet_b30 > logs/infer_thebe_pair.log 2>&1
