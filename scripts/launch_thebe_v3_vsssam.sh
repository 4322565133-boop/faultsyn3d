#!/bin/bash
# Scaled-down VSS-SAM++ (models/vss_sam3d.py) on the Thebe v3-U protocol (uniform sampling, no foreground oversampling):
# frozen SAM-2 Hiera-tiny per vertical section + 3-D Mamba (SSD) branch + gated hybrid attention fusion + 3-D decoder.
# 17.8M params (4.5M trainable), 0.67 TFLOPs / 128^3.  4 GPUs, 100 epochs, whole-validation IoU every epoch, effective batch 8.
#   1) vsssam_tiny          : full model
#   2) vsssam_tiny_nomamba  : ablation, SAM branch + decoder only (no Mamba branch, no fusion)
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 3 --per-rank 2 --accum 1 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3_vsssam_tiny runs/thebe_spatial_v3_vsssam_tiny_nomamba \
    --labels "3D UNet-L 9.6M 1.54T (v3-U)" "3D MaxViT-tiny 40.2M 0.77T (v3-U)" "VSS-SAM (SAM2-Hiera-tiny frozen + 3D Mamba + gated fusion) 17.8M/4.5M trainable 0.67T" "VSS-SAM without Mamba branch" \
    --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "launch_thebe_v3_vsssa[m].sh" > logs/live_iou_thebe.log 2>&1 &
run 29903 --run runs/thebe_spatial_v3_vsssam_tiny         --variant vsssam_tiny         > logs/train_v3_vsssam_tiny.log 2>&1
run 29904 --run runs/thebe_spatial_v3_vsssam_tiny_nomamba --variant vsssam_tiny_nomamba > logs/train_v3_vsssam_tiny_nomamba.log 2>&1
