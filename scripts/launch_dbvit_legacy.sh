#!/bin/bash
# UNet-L stopped by user at epoch 55 (val IoU 0.819 > UNet-S final).  Order now:
# 1) test-set number for the UNet-L checkpoint we have  2) DoubleBlock-ViT on the archived dataset
# (same 880/110/110 split as the archived MaxViT 0.611 run, old recipe) + live curve  3) its legacy test
# 4) dbvit_frs on v2  5) six-model v2 test table
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
ARCH_HIST=/hdd1/hukaixiao/projects/_archive_20260908_faultsyn_stage1/runs/maxvit_faultvitnet_fullvol_200ep/history.csv

CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/maxvit3d_unet_l_ddp > logs/eval_unet_l_partial.log 2>&1

$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29664 \
    train/train_old_recipe_ddp.py --variant dbvit --tag dbvit_legacy_ddp --legacy --per-rank 1 --accum 2 > logs/train_dbvit_legacy_ddp.log 2>&1 &
TPID=$!
sleep 90
$PY qc/live_curves.py --old $ARCH_HIST --new runs/maxvit3d_dbvit_legacy_ddp --out runs/maxvit3d_dbvit_legacy_ddp/live_curves.png \
    --labels "MaxViT (archived run, test 0.611)" "DoubleBlock-ViT (legacy data)" --watch dbvit_legacy_ddp > logs/live_dbvit_legacy.log 2>&1 &
wait $TPID
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --legacy --runs runs/maxvit3d_dbvit_legacy_ddp > logs/eval_dbvit_legacy.log 2>&1

$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29665 \
    train/train_old_recipe_ddp.py --variant dbvit_frs --tag dbvit_frs_ddp --per-rank 1 --accum 2 > logs/train_dbvit_frs_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/unet3d_v2_oldrecipe runs/maxvit3d_unet_l_ddp runs/resnet3d_v2_oldrecipe runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_dbvit_ddp runs/maxvit3d_dbvit_frs_ddp > logs/eval_v2_six.log 2>&1
