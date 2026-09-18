#!/bin/bash
# Continues after the running unet_l_dicesq job:  unet_lg + new loss -> UNet-L old-loss resume -> UNet-L Lovasz -> tables
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py "${@:2}"; }
while pgrep -f "tag unet_l_dicesq " >/dev/null; do sleep 60; done

run 29704 --variant unet_lg --tag unet_lg_dicesq --per-rank 2 --loss dicesq_focal --select iou > logs/train_unet_lg_dicesq.log 2>&1
run 29702 --variant unet_l --tag unet_l_ddp --per-rank 2 --resume > logs/train_unet_l_ddp_resume.log 2>&1
run 29703 --variant unet_l --tag unet_l_lovasz --per-rank 2 --loss bce_lovasz --select iou > logs/train_unet_l_lovasz.log 2>&1

CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_l_dicesq runs/maxvit3d_unet_lg_dicesq runs/maxvit3d_unet_l_lovasz runs/unet3d_v2_oldrecipe runs/maxvit3d_v2_oldrecipe_all3 > logs/eval_loss_ablation.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/analyze_loss_vs_iou.py --gpu 0 --out logs/loss_vs_iou_ablation.json --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_l_dicesq runs/maxvit3d_unet_lg_dicesq runs/maxvit3d_unet_l_lovasz > logs/analyze_loss_ablation.txt 2>&1
