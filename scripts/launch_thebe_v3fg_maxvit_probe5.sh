#!/bin/bash
# v3-FG50 probe, MaxViT-tiny only: first 5 epochs (100-epoch lr schedule, comparable per epoch with the v3-U reference),
# --fg-frac 0.5 --fg-min 0.005, evaluation unchanged.  Resumable to 100 epochs with --resume --stop-after 100.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3fg_unet_l runs/thebe_spatial_v3fg_maxvit_tiny \
    --labels "UNet-L v3-U (ref)" "MaxViT-tiny v3-U (ref)" "UNet-L v3-FG50 (stopped @2)" "MaxViT-tiny v3-FG50" \
    --out runs/thebe_live_iou.png --every 30 --ymin 0 --xmax 100 --watch "launch_thebe_v3fg_maxvit_probe[5].sh" > logs/live_iou_thebe.log 2>&1 &
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29907 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 5 --full-every 1 --workers 4 --per-rank 2 --accum 1 --fg-frac 0.5 --fg-min 0.005 \
    --run runs/thebe_spatial_v3fg_maxvit_tiny --variant baseline > logs/train_v3fg_maxvit_tiny.log 2>&1
