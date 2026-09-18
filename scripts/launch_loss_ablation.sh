#!/bin/bash
# Loss ablation on UNet-L (base 42), everything else = old recipe.  Selection by val IoU for the new losses.
#   1) dicesq_focal : 0.6 * V-Net squared-denominator Dice + 0.4 * focal   (floor-immune; minimal change)
#   2) UNet-L       : resume the dice_focal baseline from epoch 56 to 200
#   3) bce_lovasz   : alpha-weighted BCE + Lovasz hinge                    (direct IoU surrogate)
#   4) test table + loss decomposition on the three
cd /hdd1/hukaixiao/projects/faultsyn3d
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OMP_NUM_THREADS=8
run() { $PY -m torch.distributed.run --nproc_per_node 4 --master_addr 127.0.0.1 --master_port $1 train/train_old_recipe_ddp.py "${@:2}"; }

run 29701 --variant unet_l --tag unet_l_dicesq --per-rank 2 --loss dicesq_focal --select iou > logs/train_unet_l_dicesq.log 2>&1 &
TP=$!; sleep 120
$PY qc/live_curves.py --old runs/maxvit3d_unet_l_ddp --new runs/maxvit3d_unet_l_dicesq --out runs/maxvit3d_unet_l_dicesq/live_curves.png \
    --labels "UNet-L dice+focal" "UNet-L squared-dice+focal" --watch "tag unet_l_dicesq" --every 90 > logs/live_unet_l_dicesq.log 2>&1 &
wait $TP

run 29702 --variant unet_l --tag unet_l_ddp --per-rank 2 --resume > logs/train_unet_l_ddp_resume.log 2>&1

run 29703 --variant unet_l --tag unet_l_lovasz --per-rank 2 --loss bce_lovasz --select iou > logs/train_unet_l_lovasz.log 2>&1 &
TP=$!; sleep 120
$PY qc/live_curves.py --old runs/maxvit3d_unet_l_ddp --new runs/maxvit3d_unet_l_lovasz --out runs/maxvit3d_unet_l_lovasz/live_curves.png \
    --labels "UNet-L dice+focal" "UNet-L BCE+Lovasz" --watch "tag unet_l_lovasz" --every 90 > logs/live_unet_l_lovasz.log 2>&1 &
wait $TP

CUDA_VISIBLE_DEVICES=0 $PY train/evaluate.py --gpu 0 --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_l_dicesq runs/maxvit3d_unet_l_lovasz runs/unet3d_v2_oldrecipe runs/maxvit3d_v2_oldrecipe_all3 > logs/eval_loss_ablation.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY train/analyze_loss_vs_iou.py --gpu 0 --out logs/loss_vs_iou_ablation.json --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_l_dicesq runs/maxvit3d_unet_l_lovasz > logs/analyze_loss_ablation.txt 2>&1
