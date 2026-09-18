#!/bin/bash
# DoubleBlock-ViT (faithful port, 7.87M) under the Thebe spatial v3 protocol: 100 epochs, full validation every epoch,
# effective batch 8 (per-rank 1 x accum 2: the model needs ~7.8 GB per volume).
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_spatial_v3_dbvit runs/thebe_spatial_v3_context_both_stopped15 \
    --labels "UNet-L 9.6M" "MaxViT-tiny all3 40.2M" "DoubleBlock-ViT 7.9M" "Context-Grid (stopped @15)" --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "v3_dbvit.sh" > logs/live_iou_thebe.log 2>&1 &
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29894 -- train/train_thebe_spatial_ddp.py --epochs 100 --stop-after 100 --full-every 1 --workers 3 \
    --run runs/thebe_spatial_v3_dbvit --variant dbvit --per-rank 1 --accum 2 > logs/train_v3_dbvit.log 2>&1
