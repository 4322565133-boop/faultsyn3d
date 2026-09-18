#!/bin/bash
# Thebe spatial v3 protocol, whole-validation IoU EVERY epoch (2433 cores), 100 epochs, effective batch 8, from scratch.
#   1) UNet-L 9.6M (per-rank 1 x accum 2)   2) MaxViT-tiny all3 40.2M (per-rank 2)   3) Context-Grid MaxViT both 32.4M (per-rank 2)
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 3 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3_context_both \
    --labels "UNet-L 9.6M" "MaxViT-tiny all3 40.2M" "Context-Grid MaxViT (new, 32.4M)" --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "v3_models.sh" > logs/live_iou_thebe.log 2>&1 &
run 29891 --run runs/thebe_spatial_v3_unet_l       --variant unet_l       --per-rank 2 --accum 1 > logs/train_v3_unet_l.log 2>&1
run 29892 --run runs/thebe_spatial_v3_maxvit_tiny  --variant baseline     --per-rank 2 --accum 1 > logs/train_v3_maxvit_tiny.log 2>&1
run 29893 --run runs/thebe_spatial_v3_context_both --variant context_both --per-rank 2 --accum 1 > logs/train_v3_context_both.log 2>&1
