"""Evaluate trained baselines on the held-out test split and tabulate.

    python train/evaluate.py --runs runs/unet3d_v1 runs/resnet3d_v1 runs/maxvit3d_v1

Writes <run>/test.json for each run and prints one comparison table.  The test
split (replicate k >= 180) is never touched during training or model
selection, which used val (160 <= k < 180) only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build                              # noqa: E402
from train.dataset import FaultVolumes, CATEGORIES    # noqa: E402
from train.dataset_legacy import LegacyVolumes, LEGACY_CATEGORIES   # noqa: E402
from train.dataset_wu import WuXYC, WU_CATEGORIES                   # noqa: E402
from train.train import evaluate                      # noqa: E402


class LegacyXYC(torch.utils.data.Dataset):
    """LegacyVolumes returns the 7-tuple the DDP trainer wants; evaluate() wants (x, y, cat)."""
    def __init__(self, split):
        self.ds = LegacyVolumes(split, train=False)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        x, y, c, *_ = self.ds[i]
        return x, y, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default="data/dataset_reproduction_v2")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint", default="best.pt", choices=["best.pt", "best_iou.pt", "best_loss.pt"])
    p.add_argument("--wu", action="store_true", help="data/wu2019_1000 test split (100 volumes, noise-level categories)")
    p.add_argument("--legacy", action="store_true",
                   help="archived faults6_stratified_1100 test split (110 volumes, 6 categories)")
    a = p.parse_args()
    if a.wu and a.legacy:
        p.error("--wu and --legacy are mutually exclusive")
    device = torch.device(f"cuda:{a.gpu}")
    cats = WU_CATEGORIES if a.wu else (LEGACY_CATEGORIES if a.legacy else CATEGORIES)
    te = WuXYC("test") if a.wu else (LegacyXYC("test") if a.legacy else FaultVolumes(a.data, "test"))
    tl = DataLoader(te, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    rows = []
    for r in a.runs:
        r = Path(r)
        ck = torch.load(r / a.checkpoint, map_location="cpu", weights_only=False)  # our own checkpoint
        name = ck["args"]["model"]
        mkw = {}
        for kv in ck["args"].get("mkw", []):          # rebuild the exact variant that was trained
            k, v = kv.split("=", 1)
            for cast in (int, float):
                try: v = cast(v); break
                except ValueError: pass
            mkw[k] = v
        if "full_res_skip" in mkw: mkw["full_res_skip"] = bool(mkw["full_res_skip"])
        m = build(name, **mkw).to(device); m.load_state_dict(ck["model"])
        res = evaluate(m, tl, device, amp=True, categories=cats)
        dataset_id = "wu2019_1000" if a.wu else ("legacy" if a.legacy else Path(a.data).name)
        res.update(model=name, run=r.name, epoch=ck["epoch"], n_test=len(te),
                   dataset=dataset_id, split="test", checkpoint=a.checkpoint,
                   selection=ck.get("selection", ck["args"].get("select", "historical_val_loss")), threshold=0.5)
        dest = r / "eval" / dataset_id / "test"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (Path(a.checkpoint).stem + ".json")).write_text(json.dumps(res, indent=2))
        rows.append(res)

    short = {"en_echelon": "enech", "horsetail": "horse", "negative_flower": "negfl",
             "positive_flower": "posfl", "listric_assemblage": "listr", "intersecting_conjugate": "conj",
             "noise_low": "nz_lo", "noise_mid": "nz_mid", "noise_high": "nz_hi"}
    print(f'{"run":28s} {"ep":>3s} {"IoU":>6s} {"AP":>6s} {"Dice":>6s} {"Prec":>6s} {"Rec":>6s} | IoU: '
          + " ".join(f"{short[c]:>6s}" for c in cats) + " | AP: " + " ".join(f"{short[c]:>6s}" for c in cats))
    print("-" * 150)
    for res in rows:
        print(f'{res["run"]:28s} {res["epoch"]:3d} {res["iou"]:6.3f} {res["ap"]:6.3f} {res["dice"]:6.3f} '
              f'{res["precision"]:6.3f} {res["recall"]:6.3f} |      '
              + " ".join(f'{res[f"iou_{c}"]:6.3f}' for c in cats) + " |     "
              + " ".join(f'{res[f"ap_{c}"]:6.3f}' for c in cats))


if __name__ == "__main__":
    main()
