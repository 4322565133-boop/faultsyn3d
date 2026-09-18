"""Why can a model have a lower dice+focal loss but a lower IoU?  Decompose on the val split.

    python train/analyze_loss_vs_iou.py --runs runs/maxvit3d_unet_l_ddp runs/compact_maxvit_v2_m0_20260916

Per run: loss parts (soft dice / focal / plain BCE), hard IoU at 0.5 and at the best
threshold, AP, probability sharpness on TP / FP / FN voxels, calibration (ECE) and
the soft-vs-hard dice gap.  Everything is voxel-pooled over the 100 val volumes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build                                   # noqa: E402
from train.dataset import FaultVolumes, CATEGORIES         # noqa: E402

THR = np.round(np.arange(0.05, 0.96, 0.05), 2)
NB = 2000


def load(run, device):
    ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    mkw = {}
    for kv in ck["args"].get("mkw", []):
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try: v = cast(v); break
            except ValueError: pass
        mkw[k] = v
    if "full_res_skip" in mkw: mkw["full_res_skip"] = bool(mkw["full_res_skip"])
    if isinstance(mkw.get("blocks"), str): mkw["blocks"] = tuple(int(t) for t in mkw["blocks"].split(","))
    m = build(ck["args"]["model"], **mkw).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["epoch"]


@torch.no_grad()
def run_one(model, loader, device):
    s = dict(dice_num=0., dice_den=0., dice_sum=0., focal=0., bce=0., n=0,
             tp=np.zeros(len(THR)), fp=np.zeros(len(THR)), fn=np.zeros(len(THR)),
             hpos=np.zeros(NB), hneg=np.zeros(NB),
             p_tp=0., p_fp=0., p_fn=0., n_tp=0, n_fp=0, n_fn=0,
             fn_conf=0, fn_near=0, unc=0, tot=0,
             ece_conf=np.zeros(10), ece_acc=np.zeros(10), ece_n=np.zeros(10),
             cat_tp=np.zeros(len(CATEGORIES)), cat_fn=np.zeros(len(CATEGORIES)), cat_fp=np.zeros(len(CATEGORIES)),
             cat_fn_conf=np.zeros(len(CATEGORIES)), cat_p_tp=np.zeros(len(CATEGORIES)))
    for x, y, c in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(x)
        logit = logit.float(); p = torch.sigmoid(logit); yb = y > .5
        # loss parts exactly as train_old_recipe.dice_focal_loss (per-volume soft dice, voxel-mean focal)
        inter = (p * y).sum(); den = p.sum() + y.sum()
        s["dice_sum"] += float(1 - (2 * inter + 1e-6) / (den + 1e-6))
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        pt = torch.where(yb, p, 1 - p); at = torch.where(yb, .75, .25)
        s["focal"] += float((at * (1 - pt) ** 2 * bce).mean()); s["bce"] += float(bce.mean()); s["n"] += 1
        for i, t in enumerate(THR):
            pr = p > t
            s["tp"][i] += (pr & yb).sum().item(); s["fp"][i] += (pr & ~yb).sum().item(); s["fn"][i] += (~pr & yb).sum().item()
        pb = (p * (NB - 1)).round().long().clamp_(0, NB - 1).flatten(); m = yb.flatten()
        s["hpos"] += torch.bincount(pb[m], minlength=NB).cpu().numpy(); s["hneg"] += torch.bincount(pb[~m], minlength=NB).cpu().numpy()
        pr = p > .5; tp, fp, fn = pr & yb, pr & ~yb, ~pr & yb
        s["p_tp"] += float(p[tp].sum()); s["p_fp"] += float(p[fp].sum()); s["p_fn"] += float(p[fn].sum())
        s["n_tp"] += int(tp.sum()); s["n_fp"] += int(fp.sum()); s["n_fn"] += int(fn.sum())
        s["fn_conf"] += int((fn & (p < .05)).sum()); s["fn_near"] += int((fn & (p >= .3)).sum())
        s["unc"] += int(((p > .05) & (p < .95)).sum()); s["tot"] += p.numel()
        conf = torch.where(pr, p, 1 - p); acc = (pr == yb).float()
        b = (conf * 10).long().clamp_(0, 9).flatten()
        s["ece_conf"] += torch.bincount(b, weights=conf.flatten(), minlength=10).cpu().numpy()
        s["ece_acc"] += torch.bincount(b, weights=acc.flatten(), minlength=10).cpu().numpy()
        s["ece_n"] += torch.bincount(b, minlength=10).cpu().numpy()
        k = int(c[0]); s["cat_tp"][k] += int(tp.sum()); s["cat_fn"][k] += int(fn.sum()); s["cat_fp"][k] += int(fp.sum())
        s["cat_fn_conf"][k] += int((fn & (p < .05)).sum()); s["cat_p_tp"][k] += float(p[tp].sum())
    return s


def summarize(s):
    n = s["n"]
    dice = s["dice_sum"] / n; focal = s["focal"] / n; bce = s["bce"] / n
    iou = s["tp"] / np.maximum(s["tp"] + s["fp"] + s["fn"], 1)
    i50 = int(np.argmin(np.abs(THR - .5))); ib = int(np.argmax(iou))
    hard_dice = 2 * s["tp"][i50] / max(2 * s["tp"][i50] + s["fp"][i50] + s["fn"][i50], 1)
    cp = np.cumsum(s["hpos"][::-1]); cn = np.cumsum(s["hneg"][::-1])
    P = cp / np.maximum(cp + cn, 1); R = cp / max(s["hpos"].sum(), 1); Rp = np.concatenate([[0.], R[:-1]])
    ap = float(np.sum((R - Rp) * P))
    ece = float(np.sum(np.abs(s["ece_conf"] - s["ece_acc"])) / max(s["ece_n"].sum(), 1))
    out = dict(loss=.6 * dice + .4 * focal, dice_soft=dice, focal=focal, bce=bce, dice_hard=1 - hard_dice,
               iou50=float(iou[i50]), P50=float(s["tp"][i50] / max(s["tp"][i50] + s["fp"][i50], 1)),
               R50=float(s["tp"][i50] / max(s["tp"][i50] + s["fn"][i50], 1)),
               best_thr=float(THR[ib]), iou_best=float(iou[ib]), ap=ap,
               p_tp=s["p_tp"] / max(s["n_tp"], 1), p_fp=s["p_fp"] / max(s["n_fp"], 1), p_fn=s["p_fn"] / max(s["n_fn"], 1),
               fn_conf_frac=s["fn_conf"] / max(s["n_fn"], 1), fn_near_frac=s["fn_near"] / max(s["n_fn"], 1),
               unc_frac=s["unc"] / s["tot"], ece=ece,
               iou_curve={f"{t:.2f}": float(v) for t, v in zip(THR, iou)})
    for k, cat in enumerate(CATEGORIES):
        out[f"R_{cat}"] = s["cat_tp"][k] / max(s["cat_tp"][k] + s["cat_fn"][k], 1)
        out[f"fnconf_{cat}"] = s["cat_fn_conf"][k] / max(s["cat_fn"][k], 1)
        out[f"ptp_{cat}"] = s["cat_p_tp"][k] / max(s["cat_tp"][k], 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default="data/dataset_reproduction_v2")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="logs/loss_vs_iou.json")
    a = p.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    dl = DataLoader(FaultVolumes(a.data, "val"), batch_size=1, num_workers=2, pin_memory=True)
    res = {}
    for r in a.runs:
        m, ep = load(r, device); res[Path(r).name] = dict(epoch=ep, **summarize(run_one(m, dl, device)))
        del m; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))
    keys = ["epoch", "loss", "dice_soft", "dice_hard", "focal", "bce", "iou50", "P50", "R50", "best_thr", "iou_best", "ap",
            "p_tp", "p_fp", "p_fn", "fn_conf_frac", "fn_near_frac", "unc_frac", "ece"]
    print(f'{"":14s}' + "".join(f"{Path(r).name[:22]:>24s}" for r in a.runs))
    for k in keys:
        print(f"{k:14s}" + "".join(f'{res[Path(r).name][k]:24.4f}' if k != "epoch" else f'{res[Path(r).name][k]:24d}' for r in a.runs))
    print("\nrecall / confidently-missed FN share (p<.05) / mean p on TP, per category")
    for cat in CATEGORIES:
        print(f"{cat[:14]:14s}" + "".join(f'{res[Path(r).name][f"R_{cat}"]:8.3f} {res[Path(r).name][f"fnconf_{cat}"]:7.3f} {res[Path(r).name][f"ptp_{cat}"]:7.3f}' for r in a.runs))
    print("\nIoU vs threshold")
    for t in ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]:
        print(f"thr {t:5s}" + "".join(f'{res[Path(r).name]["iou_curve"][t]:24.4f}' for r in a.runs))


if __name__ == "__main__":
    main()
