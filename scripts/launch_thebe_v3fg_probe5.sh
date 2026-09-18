#!/bin/bash
# v3-FG50 probe: UNet-L and MaxViT-tiny, first 5 epochs only (same 100-epoch lr schedule as the v3-U references, so the
# per-epoch curves are comparable), foreground oversampling --fg-frac 0.5 --fg-min 0.005, evaluation unchanged.
# Run dirs are the final v3-FG50 names; both can be continued to 100 epochs later with --resume --stop-after 100.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 5 --full-every 1 --workers 4 --per-rank 2 --accum 1 --fg-frac 0.5 --fg-min 0.005 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3fg_unet_l runs/thebe_spatial_v3fg_maxvit_tiny \
    --labels "UNet-L v3-U (ref)" "MaxViT-tiny v3-U (ref)" "UNet-L v3-FG50" "MaxViT-tiny v3-FG50" \
    --out runs/thebe_live_iou.png --every 30 --ymin 0 --xmax 100 --watch "launch_thebe_v3fg_probe[5].sh" > logs/live_iou_thebe.log 2>&1 &
run 29905 --run runs/thebe_spatial_v3fg_unet_l      --variant unet_l   > logs/train_v3fg_unet_l.log 2>&1
run 29906 --run runs/thebe_spatial_v3fg_maxvit_tiny --variant baseline > logs/train_v3fg_maxvit_tiny.log 2>&1
