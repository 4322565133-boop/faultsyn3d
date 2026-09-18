#!/bin/bash
# Thebe spatial v3 protocol, 4 GPUs, whole-validation IoU every epoch, 100 epochs, effective batch 8 (per-rank 2), from scratch:
#   1) unet_lg_b30 : U-Net base30 + MaxViT-style attention at 32^3 (block+grid) and 16^3 (block+global), 6.1M, 0.81 TF (parity with MaxViT-tiny)
#   2) unet_b30    : its pure-conv twin, 4.9M, 0.79 TF (completes the compute-parity trio on Thebe)
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 3 --per-rank 2 --accum 1 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3_unet_lg_b30 runs/thebe_spatial_v3_unet_b30 \
    --labels "UNet-L 9.6M 1.54T" "MaxViT-tiny 40.2M 0.77T" "UNet+attn b30 (unet_lg) 6.1M 0.81T" "UNet b30 4.9M 0.79T" \
    --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "launch_thebe_v3_unet_l[g].sh" > logs/live_iou_thebe.log 2>&1 &
run 29897 --run runs/thebe_spatial_v3_unet_lg_b30 --variant unet_lg_b30 > logs/train_v3_unet_lg_b30.log 2>&1
run 29898 --run runs/thebe_spatial_v3_unet_b30    --variant unet_b30    > logs/train_v3_unet_b30.log 2>&1
