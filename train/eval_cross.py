"""Zero-shot cross-dataset evaluation of v2-trained checkpoints.

    python train/eval_cross.py --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_v2_oldrecipe_all3 ...

Targets (no fine-tuning, same per-volume standardisation as training):
  faultseg3d : Wu et al. 2019 public synthetic validation set, 20 x 128^3 (.dat), thick labels (~7.5 %)
  legacy     : archived faults6_stratified_1100 test split, 110 x 128^3, 6 classes, 3 noise modes
Prints voxel IoU / AP / P / R and 2-voxel-tolerance P / R (labels and predictions dilated by 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.analyze_loss_vs_iou import load                          # noqa: E402
from train.dataset_legacy import LegacyVolumes                     # noqa: E402

FS3D = Path("/hdd1/hukaixiao/projects/FAULTSEG3D/data/wangjing/validation")


class FaultSeg3DVal(Dataset):
    def __init__(self):
        self.ids = sorted(int(p.stem) for p in (FS3D / "seis").glob("*.dat"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i]
        # released .dat volumes are [x, y, z] (depth last); this project is z-first
        x = np.fromfile(FS3D / "seis" / f"{k}.dat", dtype=np.float32).reshape(128, 128, 128).transpose(2, 1, 0)
        y = np.fromfile(FS3D / "fault" / f"{k}.dat", dtype=np.float32).reshape(128, 128, 128).transpose(2, 1, 0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x[None].copy()), torch.from_numpy((y > .5).astype(np.float32)[None]), 0


class LegacyTest(Dataset):
    def __init__(self):
        self.ds = LegacyVolumes("test", train=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, c, *_ = self.ds[i]
        return x, y, c


@torch.no_grad()
def run(model, loader, device, NB=2000):
    tp = fp = fn = 0.0; ttp = tfp = tfn = 0.0; matched_gt = 0.0
    hpos = np.zeros(NB); hneg = np.zeros(NB)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x).float())
        yb = y > .5; pr = p > .5
        tp += (pr & yb).sum().item(); fp += (pr & ~yb).sum().item(); fn += (~pr & yb).sum().item()
        pb = (p * (NB - 1)).round().long().clamp_(0, NB - 1).flatten(); m = yb.flatten()
        hpos += torch.bincount(pb[m], minlength=NB).cpu().numpy(); hneg += torch.bincount(pb[~m], minlength=NB).cpu().numpy()
        yd = F.max_pool3d(y, 5, 1, 2) > .5; pd = F.max_pool3d(pr.float(), 5, 1, 2) > .5     # 2-voxel tolerance
        ttp += (pr & yd).sum().item(); tfp += (pr & ~yd).sum().item(); tfn += (~pd & yb).sum().item()
        matched_gt += (pd & yb).sum().item()
    cp = np.cumsum(hpos[::-1]); cn = np.cumsum(hneg[::-1])
    P = cp / np.maximum(cp + cn, 1); R = cp / max(hpos.sum(), 1); Rp = np.concatenate([[0.], R[:-1]])
    return dict(iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                ap=float(np.sum((R - Rp) * P)), tol2_precision=ttp / max(ttp + tfp, 1), tol2_recall=matched_gt / max(matched_gt + tfn, 1))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True); a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--out", default="logs/eval_cross.json")
    a.add_argument("--sets", nargs="+", default=["faultseg3d", "legacy"])
    a = a.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    sets = {"faultseg3d": DataLoader(FaultSeg3DVal(), batch_size=1, num_workers=2),
            "legacy": DataLoader(LegacyTest(), batch_size=1, num_workers=2)}
    sets = {k: v for k, v in sets.items() if k in a.sets}
    res = {}
    for r in a.runs:
        m, ep = load(r, device); name = Path(r).name; res[name] = {"epoch": ep}
        for k, dl in sets.items():
            res[name][k] = run(m, dl, device)
        del m; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k in sets:
        print(f"\n== {k}")
        print(f'{"run":30s} {"ep":>3s} {"IoU":>6s} {"AP":>6s} {"P":>6s} {"R":>6s} {"tol2P":>6s} {"tol2R":>6s}')
        for name, v in res.items():
            s = v[k]; print(f'{name:30s} {v["epoch"]:3d} {s["iou"]:6.3f} {s["ap"]:6.3f} {s["precision"]:6.3f} {s["recall"]:6.3f} {s["tol2_precision"]:6.3f} {s["tol2_recall"]:6.3f}')


if __name__ == "__main__":
    main()
