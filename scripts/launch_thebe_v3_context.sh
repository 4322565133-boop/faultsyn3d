#!/bin/bash
# starts the Context-Grid run as soon as the sealed-test evaluation releases GPU 0
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
while pgrep -f "eval_thebe_spatia[l].py" >/dev/null; do sleep 10; done
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3_context_both \
    --labels "UNet-L 9.6M" "MaxViT-tiny all3 40.2M" "Context-Grid MaxViT (new, 32.4M)" --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "v3_context.sh" > logs/live_iou_thebe.log 2>&1 &
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29893 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 3 \
    --run runs/thebe_spatial_v3_context_both --variant context_both --per-rank 2 --accum 1 > logs/train_v3_context_both.log 2>&1
