#!/bin/bash
# DoubleBlock-ViT U-Net (faithful port), our data + old recipe on 4 GPUs (1/rank x accum 2 = global 8), 200 ep.
# Then the full-resolution-skip variant.  Both evaluated against the three baselines on the test split.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29621 \
    train/train_old_recipe_ddp.py --variant dbvit --tag dbvit_ddp --per-rank 1 --accum 2 > logs/train_dbvit_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/unet3d_v2_oldrecipe runs/resnet3d_v2_oldrecipe runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_dbvit_ddp > logs/eval_dbvit.log 2>&1
$PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29622 \
    train/train_old_recipe_ddp.py --variant dbvit_frs --tag dbvit_frs_ddp --per-rank 1 --accum 2 > logs/train_dbvit_frs_ddp.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/unet3d_v2_oldrecipe runs/resnet3d_v2_oldrecipe runs/maxvit3d_v2_oldrecipe_all3 runs/maxvit3d_dbvit_ddp runs/maxvit3d_dbvit_frs_ddp > logs/eval_dbvit_both.log 2>&1
