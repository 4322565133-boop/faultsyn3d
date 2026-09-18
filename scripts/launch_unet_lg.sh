#!/bin/bash
# UNet-L + MaxViT-style attention (models/unet_maxvit3d.py), old recipe, 4 GPUs.
#   1) unet_lg      : UNet-L kept, block+grid attention after enc3 (32^3), block+global after mid (16^3)  11.94M
#   2) UNet-L       : resume runs/maxvit3d_unet_l_ddp from epoch 56 to 200 (the baseline to beat)         9.64M
#   3) unet_lg_rep  : same attention, but the second 3^3 conv of enc3/mid removed (conv -> attention)     8.13M
#   4) test table
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py "${@:2}"; }

run 29691 --variant unet_lg --tag unet_lg_ddp --per-rank 2 > logs/train_unet_lg_ddp.log 2>&1 &
TP=$!; sleep 120
$PY qc/live_curves.py --old runs/maxvit3d_unet_l_ddp --new runs/maxvit3d_unet_lg_ddp --out runs/maxvit3d_unet_lg_ddp/live_curves.png \
    --labels "UNet-L (9.6M)" "UNet-L + MaxViT attn (11.9M)" --watch "variant unet_lg " --every 90 > logs/live_unet_lg.log 2>&1 &
wait $TP

run 29692 --variant unet_l --tag unet_l_ddp --per-rank 2 --resume > logs/train_unet_l_ddp_resume.log 2>&1 &
TP=$!; sleep 120
$PY qc/live_curves.py --old runs/maxvit3d_unet_lg_ddp --new runs/maxvit3d_unet_l_ddp --out runs/maxvit3d_unet_l_ddp/live_curves_resume.png \
    --labels "UNet-L + MaxViT attn (11.9M)" "UNet-L (9.6M, resumed @57)" --watch "variant unet_l " --every 90 > logs/live_unet_l_resume.log 2>&1 &
wait $TP

run 29693 --variant unet_lg_rep --tag unet_lg_rep_ddp --per-rank 2 > logs/train_unet_lg_rep_ddp.log 2>&1 &
TP=$!; sleep 120
$PY qc/live_curves.py --old runs/maxvit3d_unet_l_ddp --new runs/maxvit3d_unet_lg_rep_ddp --out runs/maxvit3d_unet_lg_rep_ddp/live_curves.png \
    --labels "UNet-L (9.6M)" "UNet-L conv->attn replace (8.1M)" --watch "variant unet_lg_rep" --every 90 > logs/live_unet_lg_rep.log 2>&1 &
wait $TP

CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/unet3d_v2_oldrecipe runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_lg_ddp runs/maxvit3d_unet_lg_rep_ddp runs/maxvit3d_v2_oldrecipe_all3 runs/compact_maxvit_v2_m0_20260916 > logs/eval_unet_lg.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/analyze_runs.py --gpu 0 --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_lg_ddp runs/maxvit3d_unet_lg_rep_ddp > logs/analyze_unet_lg.txt 2>&1
