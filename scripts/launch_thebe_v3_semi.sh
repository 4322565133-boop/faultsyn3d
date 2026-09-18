#!/bin/bash
# Semi-supervised pilot on Thebe (spatial v3 protocol): MaxViT-tiny with 10 % of the train-region labels
# (2 blocks of 159 sections along axis 1: [714,873) and [2301,2460)), 60 epochs x 1200 draws, effective batch 8,
# whole-validation IoU every epoch.  1) supervised only  2) mean-teacher semi-supervision (EMA 0.99, confident
# pseudo-labels, label-density-guided copy-paste, spatial-masking consistency, lambda ramp 20 epochs).
# NOTE 2026-09-18 11:20: per the user, two GPUs only (0 and 1): per-rank 1 x accum 4 x 2 ranks = effective batch 8 (the recipe's).
cd /hdd1/hukaixiao/projects/faultsyn3d
export CUDA_VISIBLE_DEVICES=0,1
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=4
run() { $PY -m torch.distributed.run --nproc_per_node 2 --master_addr 127.0.0.1 --master_port $1 -- train/train_thebe_spatial_ddp.py \
        --variant baseline --label-frac 0.1 --label-blocks 2 --samples 1200 --epochs 60 --stop-after 60 --full-every 1 --workers 3 --per-rank 1 --accum 4 "${@:2}"; }
$PY qc/live_iou.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny runs/thebe_v3_semi/maxvit_tiny_lab10_sup runs/thebe_v3_semi/maxvit_tiny_lab10_semi \
    --labels "UNet-L 100% labels (ref, 2400/ep)" "MaxViT-tiny 100% labels (ref, 2400/ep)" "MaxViT-tiny 10% labels, supervised (1200/ep)" "MaxViT-tiny 10% labels, semi-supervised (1200/ep)" \
    --out runs/thebe_live_iou.png --every 60 --ymin 0 --xmax 100 --watch "launch_thebe_v3_sem[i].sh" > logs/live_iou_thebe.log 2>&1 &
run 29895 --run runs/thebe_v3_semi/maxvit_tiny_lab10_sup         > logs/train_v3_semi_lab10_sup.log 2>&1
run 29896 --run runs/thebe_v3_semi/maxvit_tiny_lab10_semi --semi > logs/train_v3_semi_lab10_semi.log 2>&1
