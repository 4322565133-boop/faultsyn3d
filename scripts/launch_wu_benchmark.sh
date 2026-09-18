#!/bin/bash
# Benchmark on data/wu2019_1000 (Wu et al. 2019 style synthetic, 800/100/100), old recipe, 4 GPUs.
# After each run: test split (evaluate.py --wu) and zero-shot on Wu's released 20 validation volumes.
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py --wu "${@:2}"; }
ev() { CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --wu --runs runs/maxvit3d_$1 > logs/eval_wu_$1.log 2>&1
       CUDA_VISIBLE_DEVICES=0 $PY train/eval_cross.py --gpu 0 --sets faultseg3d --out logs/eval_wureal_$1.json --runs runs/maxvit3d_$1 > logs/eval_wureal_$1.log 2>&1; }

run 29721 --variant unet_l      --tag wu_unet_l        --per-rank 2 > logs/train_wu_unet_l.log 2>&1;        ev wu_unet_l
run 29722 --variant unet_lg     --tag wu_unet_lg       --per-rank 2 > logs/train_wu_unet_lg.log 2>&1;       ev wu_unet_lg
run 29723 --variant unet_l      --tag wu_unet_l_dicesq --per-rank 2 --loss dicesq_focal --select iou > logs/train_wu_unet_l_dicesq.log 2>&1;  ev wu_unet_l_dicesq
run 29724 --variant unet_lg     --tag wu_unet_lg_dicesq --per-rank 2 --loss dicesq_focal --select iou > logs/train_wu_unet_lg_dicesq.log 2>&1; ev wu_unet_lg_dicesq
run 29725 --variant baseline    --tag wu_maxvit_all3   --per-rank 2 > logs/train_wu_maxvit_all3.log 2>&1;   ev wu_maxvit_all3
run 29726 --variant unet_s      --tag wu_unet_s        --per-rank 2 > logs/train_wu_unet_s.log 2>&1;        ev wu_unet_s

CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --wu --runs runs/maxvit3d_wu_unet_s runs/maxvit3d_wu_unet_l runs/maxvit3d_wu_unet_lg runs/maxvit3d_wu_unet_l_dicesq runs/maxvit3d_wu_unet_lg_dicesq runs/maxvit3d_wu_maxvit_all3 > logs/eval_wu_all.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/eval_cross.py --gpu 0 --sets faultseg3d --out logs/eval_wureal_all.json --runs runs/maxvit3d_wu_unet_s runs/maxvit3d_wu_unet_l runs/maxvit3d_wu_unet_lg runs/maxvit3d_wu_unet_l_dicesq runs/maxvit3d_wu_unet_lg_dicesq runs/maxvit3d_wu_maxvit_all3 > logs/eval_wureal_all.log 2>&1
