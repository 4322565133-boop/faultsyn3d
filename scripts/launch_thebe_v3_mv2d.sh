#!/bin/bash
# 2-D MaxViT (timm maxvit_tiny_tf_224, ImageNet init) per slice + cross-slice depth adapters + 3-D U-Net decoder
# (models/maxvit2d_adapter3d.py, 15.0M params, 0.64 TFLOPs / 128^3) on the Thebe spatial v3 protocol, 4 GPUs,
# 100 epochs, whole-validation IoU every epoch, effective batch 8 (per-rank 1 x accum 2; 7.5 GB/GPU).
#   1) mv2d_tiny_ad   : full model (adapters at all four scales)
#   2) mv2d_tiny_noad : ablation, 2-D MaxViT + 3-D decoder only (no cross-slice adapter)
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 3 --per-rank 1 --accum 2 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3_mv2d_tiny_ad runs/thebe_spatial_v3_mv2d_tiny_noad \
    --labels "3D UNet-L 9.6M 1.54T" "3D MaxViT-tiny 40.2M 0.77T" "2D MaxViT + depth adapter + 3D dec 15.0M 0.64T" "2D MaxViT + 3D dec, no adapter" \
    --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "launch_thebe_v3_mv2[d].sh" > logs/live_iou_thebe.log 2>&1 &
run 29899 --run runs/thebe_spatial_v3_mv2d_tiny_ad   --variant mv2d_tiny_ad   > logs/train_v3_mv2d_tiny_ad.log 2>&1
run 29900 --run runs/thebe_spatial_v3_mv2d_tiny_noad --variant mv2d_tiny_noad > logs/train_v3_mv2d_tiny_noad.log 2>&1
