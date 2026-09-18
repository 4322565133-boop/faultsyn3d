#!/bin/bash
# After dbvit_frs finishes: UNet-L (base 42, 9.64 M) under the same DDP old recipe, then a 5-model test table.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
while pgrep -f "train_old_recipe_ddp.py --variant dbvit_frs" >/dev/null || pgrep -f "launch_dbvit.sh" >/dev/null; do sleep 60; done
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29631 \
    train/train_old_recipe_ddp.py --variant unet_l --tag unet_l_ddp --per-rank 2 > logs/train_unet_l_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/unet3d_v2_oldrecipe runs/maxvit3d_unet_l_ddp runs/resnet3d_v2_oldrecipe runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_dbvit_ddp runs/maxvit3d_dbvit_frs_ddp > logs/eval_unet_l.log 2>&1
