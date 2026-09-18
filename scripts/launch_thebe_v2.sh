#!/bin/bash
# Thebe, sampler v2: 2400 random 128^3 cubes RE-DRAWN EVERY EPOCH from the train volume (empty cubes kept w.p. 0.7),
# stratified 600-cube val (45/15/40 none/sparse/dense, like the test grid).  Old recipe defaults, cap 100 epochs.
# Compute-parity trio at 0.77 TF: MaxViT-tiny all3 40.2M | UNet base30 4.9M | UNet+MaxViT-attn base30 6.1M.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --thebe --epochs 100 --workers 4 "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --thebe --runs runs/maxvit3d_$1 > logs/eval_thebe_$1.log 2>&1
       CUDA_VISIBLE_DEVICES=0 $PY train/infer_thebe.py --gpu 0 --runs runs/maxvit3d_$1 > logs/infer_thebe_$1.log 2>&1; }
run 29851 --variant baseline    --tag thebe2_maxvit_tiny --per-rank 2 > logs/train_thebe2_maxvit_tiny.log 2>&1; ev thebe2_maxvit_tiny
run 29852 --variant unet_b30    --tag thebe2_unet_b30    --per-rank 2 > logs/train_thebe2_unet_b30.log 2>&1;    ev thebe2_unet_b30
run 29853 --variant unet_lg_b30 --tag thebe2_unet_lg_b30 --per-rank 2 > logs/train_thebe2_unet_lg_b30.log 2>&1; ev thebe2_unet_lg_b30
CUDA_VISIBLE_DEVICES=0 $PY train/infer_thebe.py --gpu 0 --no-save --runs runs/maxvit3d_thebe2_maxvit_tiny runs/maxvit3d_thebe2_unet_b30 runs/maxvit3d_thebe2_unet_lg_b30 > logs/infer_thebe2_all.log 2>&1
