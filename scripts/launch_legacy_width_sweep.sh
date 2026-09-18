#!/bin/bash
# Capacity probe on the archived dataset: plain UNet3D at base 16/32/42/64/96/128 (1.4 / 5.6 / 9.6 / 22 / 50 / 90 M),
# 10 epochs each under the old recipe (epochs 1-10 are the warmup, identical to the 200-epoch schedule), global batch 8.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --legacy --epochs 10 "${@:2}"; }
run 29771 --variant unet_s    --tag sweep_b16  --per-rank 2           > logs/train_sweep_b16.log 2>&1
run 29772 --variant unet_b32  --tag sweep_b32  --per-rank 2           > logs/train_sweep_b32.log 2>&1
run 29773 --variant unet_l    --tag sweep_b42  --per-rank 2           > logs/train_sweep_b42.log 2>&1
run 29774 --variant unet_b64  --tag sweep_b64  --per-rank 1 --accum 2 > logs/train_sweep_b64.log 2>&1
run 29775 --variant unet_b96  --tag sweep_b96  --per-rank 1 --accum 2 > logs/train_sweep_b96.log 2>&1
run 29776 --variant unet_b128 --tag sweep_b128 --per-rank 1 --accum 2 > logs/train_sweep_b128.log 2>&1
