#!/bin/bash
# Thebe spatial v3 protocol with foreground oversampling in TRAINING only (--fg-frac 0.5 --fg-min 0.005: half of the
# 128^3 draws must contain >= 0.5 % fault voxels; evaluation unchanged: whole validation / sealed test region).
# 4 GPUs, 100 epochs, whole-validation IoU every epoch, effective batch 8.  1) MaxViT-tiny  2) UNet-L
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 4 --per-rank 2 --accum 1 --fg-frac 0.5 --fg-min 0.005 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3fg_maxvit_tiny runs/thebe_spatial_v3fg_unet_l \
    --labels "UNet-L, random sampling (ref)" "MaxViT-tiny, random sampling (ref)" "MaxViT-tiny, 50% foreground oversampling" "UNet-L, 50% foreground oversampling" \
    --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "launch_thebe_v3f[g].sh" > logs/live_iou_thebe.log 2>&1 &
run 29901 --run runs/thebe_spatial_v3fg_maxvit_tiny --variant baseline > logs/train_v3fg_maxvit_tiny.log 2>&1
run 29902 --run runs/thebe_spatial_v3fg_unet_l      --variant unet_l   > logs/train_v3fg_unet_l.log 2>&1
