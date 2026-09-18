"""The archived stage-1 recipe, unchanged, on 4 GPUs -- plus optional D2S / geometry losses.

Everything that produced runs/maxvit3d_v2_oldrecipe_all3 (test IoU 0.833) is
kept as it was, so that run remains a valid baseline for what is trained here:

  loss          0.6 * Dice + 0.4 * Focal(0.75, 2) -- the SAME function, called
                under the same fp16 autocast as train_old_recipe.py
  augmentation  rot90 / flips / blur p=0.35 / intensity / z-score (GeometryVolumes
                reproduces OldAugVolumes and transforms the geometry cache with it)
  optimiser     AdamW, wd 0.01, ONE param group at ONE lr, NO gradient clipping
  schedule      per-epoch warmup 1e-6 -> 1e-4 over 10 epochs, cosine -> 1e-7 over 200
  batch         global 8: 4 ranks x 2 per rank, DDP-averaged = mean over 8, exactly
                what 2 x accum 4 computed on one GPU
  selection     lowest val dice_focal loss (seg term only), patience 20
  AMP           fp16 autocast + GradScaler, default overflow handling

The only things that differ from the old entry are the ones needed for the
upgrade itself: the model may carry D2S fusion, and the two geometry losses may
be added on top of the seg loss (fp32, ramped in after warmup).  Both are off
by default, so `--variant baseline` reproduces the old entry on 4 GPUs.

    torchrun --nproc_per_node 4 train/train_old_recipe_ddp.py --variant full --tag full_oldrecipe_ddp
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build                                          # noqa: E402
from train.dataset import FaultVolumes, CATEGORIES                 # noqa: E402
from train.geometry_dataset import GeometryVolumes                 # noqa: E402
from train.dataset_legacy import LegacyVolumes, LEGACY_CATEGORIES   # noqa: E402
from train.dataset_wu import WuVolumes, WU_CATEGORIES               # noqa: E402
from train.dataset_thebe import ThebeCubes, ThebeRandomCubes, THEBE_CATEGORIES   # noqa: E402
from train.geometry_losses import profile_loss, trace_loss         # noqa: E402
from train.train_old_recipe import (dice_focal_loss, warmup_cosine_lr, LOSSES,   # noqa: E402
                                    OldAugVolumes)

ALL3 = dict(full_res_skip=True, drop_path_rate=0.2, img_size=128)
VARIANTS = {                       # model name, model kwargs, profile, trace
    # Experimental field-data decoder; no queued experiment is changed by these new names.
    "context_conv": ("context_grid_maxvit3d", dict(mode="conv"), False, False),
    "context_block": ("context_grid_maxvit3d", dict(mode="block"), False, False),
    "context_grid": ("context_grid_maxvit3d", dict(mode="grid"), False, False),
    "context_both": ("context_grid_maxvit3d", dict(mode="both"), False, False),
    "baseline":  ("maxvit3d",     ALL3, False, False),   # B0: all3 architecture
    "d2s":       ("maxvit3d_d2s", ALL3, False, False),
    "loss":      ("maxvit3d",     ALL3, True,  True),
    "full":      ("maxvit3d_d2s", ALL3, True,  True),    # negative result, 141 ep == baseline
    "dyn":       ("maxvit3d_dyn", ALL3, False, False),   # all3 + dynamic upsample 32^3 -> 64^3
    # DoubleBlock-ViT U-Net (Nguyen-Tat et al. 2026), faithful port of the official code
    "dbvit":     ("dbvit", dict(n_channels=16, blocks=(1, 1, 2), full_res_skip=False), False, False),
    "dbvit_frs": ("dbvit", dict(n_channels=16, blocks=(1, 1, 2), full_res_skip=True),  False, False),
    # UNet-L: the UNet-S architecture widened to base 42 -> 9.64 M params, matched to the
    # 9.74 M "U-net" baseline reported in Ma et al., TGRS 2023 (Table I).  Capacity control.
    "unet_s":    ("unet3d", dict(base=16), False, False),   # the 1.4 M baseline
    # same all3 U-Net wrapper on the smaller timm_3d MaxViT backbones (pico 9.8 M ~ UNet-L, nano 24.8 M)
    "maxvit_pico": ("maxvit3d", dict(ALL3, backbone_name="maxvit_pico_rw_256"), False, False),
    "maxvit_nano": ("maxvit3d", dict(ALL3, backbone_name="maxvit_nano_rw_256"), False, False),
    "unet_l":    ("unet3d", dict(base=42), False, False),
    # width sweep for the capacity test (params 5.6 / 22 / 50 / 90 M)
    "unet_b32":  ("unet3d", dict(base=32),  False, False),
    # compute-matched to MaxViT-tiny all3 (0.77 TFLOPs / volume): UNet base30 0.79 TF, UNet+attn base30 0.81 TF
    "unet_b30":     ("unet3d", dict(base=30), False, False),
    # compute-matched to MaxViT-pico all3 (0.40 TFLOPs / volume): UNet base21 0.39 TF, UNet+attn base21 0.40 TF
    # (head dim must be a multiple of 8 or SDPA falls back to the O(N^2) math kernel -> OOM at 16^3 global attn)
    "unet_b22":     ("unet3d", dict(base=22), False, False),                       # 2.65 M, 0.42 TF
    "unet_lg_b22":  ("unet_maxvit3d", dict(base=22, dim_head=8), False, False),    # 3.30 M, 0.44 TF
    "unet_lg_b30":  ("unet_maxvit3d", dict(base=30, dim_head=24), False, False),
    # 2-D MaxViT (timm maxvit_tiny_tf_224, ImageNet init, 4 scales) per slice + depth adapters + 3-D U-Net decoder
    "mv2d_tiny_ad":         ("maxvit2d_adapter3d", dict(backbone="maxvit_tiny_tf_224", pretrained=True, adapter=True), False, False),
    "mv2d_tiny_noad":       ("maxvit2d_adapter3d", dict(backbone="maxvit_tiny_tf_224", pretrained=True, adapter=False), False, False),
    "mv2d_tiny_ad_scratch": ("maxvit2d_adapter3d", dict(backbone="maxvit_tiny_tf_224", pretrained=False, adapter=True), False, False),
    "mv2d_tiny_ad_deep":    ("maxvit2d_adapter3d", dict(backbone="maxvit_tiny_tf_224", pretrained=True, adapter=True, adapter_levels="deep"), False, False),
    "mv2d_tiny_ad_25d":     ("maxvit2d_adapter3d", dict(backbone="maxvit_tiny_tf_224", pretrained=True, adapter=True, in_slices=3), False, False),
    "mv2d_nano_ad":         ("maxvit2d_adapter3d", dict(backbone="maxvit_nano_rw_256", pretrained=True, adapter=True), False, False),
    # VSS-SAM++ scaled down: frozen SAM-2 Hiera-tiny (2-D per slice) + 3-D Mamba (SSD) branch + gated hybrid attention fusion + 3-D decoder
    "vsssam_tiny":          ("vss_sam3d", dict(sam="sam2_hiera_tiny", pretrained=True, freeze_sam=True, mamba=True), False, False),
    "vsssam_tiny_nomamba":  ("vss_sam3d", dict(sam="sam2_hiera_tiny", pretrained=True, freeze_sam=True, mamba=False), False, False),
    "vsssam_tiny_unfrozen": ("vss_sam3d", dict(sam="sam2_hiera_tiny", pretrained=True, freeze_sam=False, mamba=True), False, False),
    "unet_b64":  ("unet3d", dict(base=64),  False, False),
    "unet_b96":  ("unet3d", dict(base=96, checkpoint_highres=True),  False, False),
    "unet_b128": ("unet3d", dict(base=128, checkpoint_highres=True), False, False),
    # UNet-L + MaxViT-style attention (block+grid at 32^3, block+global at 16^3); "rep" drops the
    # second 3^3 conv of enc3/mid so each deep stage is conv -> attention like a MaxViT stage
    "unet_lg":     ("unet_maxvit3d", dict(base=42, replace=False), False, False),
    "unet_lg_rep": ("unet_maxvit3d", dict(base=42, replace=True),  False, False),
    # same design scaled to the 40 M class: UNet base 76 (31.6 M) + block/grid attn at 32^3 (304 ch, 1.5 M)
    # + block/global attn at 16^3 (608 ch, 6.0 M) = 39.0 M, vs MaxViT-tiny all3 40.2 M
    "unet_lg_xl":  ("unet_maxvit3d", dict(base=76, dim_head=16, checkpoint_highres=True), False, False),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default="data/dataset_reproduction_v2")
    p.add_argument("--cache", default="data/geometry_model_upgrade_v1")
    p.add_argument("--epochs", type=int, default=200); p.add_argument("--patience", type=int, default=20)
    p.add_argument("--sched", choices=["cosine", "plateau"], default="cosine",
                   help="cosine: warmup then cosine over --epochs (the archived recipe); plateau: warmup then hold "
                        "the peak lr, halve it when val has not improved for --lr-patience epochs, stop on --patience "
                        "(--epochs is only a cap) -- i.e. train to convergence")
    p.add_argument("--lr-patience", type=int, default=10); p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--per-rank", type=int, default=2)
    p.add_argument("--accum", type=int, default=1, help="grad accumulation steps per rank (global = world*per_rank*accum)")
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--warmup-start-lr", type=float, default=1e-6); p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--lambda-profile", type=float, default=0.1); p.add_argument("--lambda-trace", type=float, default=0.05)
    p.add_argument("--ramp-epochs", type=int, default=20, help="geometry losses ramp 0->1 over these epochs after warmup")
    p.add_argument("--d2s-init", type=float, default=0.1, help="initial effective tanh(gamma)=tanh(eta) of D2S")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--fp32-loss", action="store_true", default=True,
                   help="compute the seg loss on logits.float() outside autocast (new-protocol default)")
    p.add_argument("--wu", action="store_true", help="train/val on data/wu2019_1000 (Wu et al. 2019 style synthetic)")
    p.add_argument("--thebe", action="store_true", help="train/val on data/thebe_cubes (Thebe field data, An et al. 2021)")
    p.add_argument("--thebe-fixed", action="store_true", help="use the fixed 2400 training cubes instead of per-epoch random cubes")
    p.add_argument("--thebe-n", type=int, default=2400, help="random training cubes drawn per epoch")
    p.add_argument("--thebe-keep-empty", type=float, default=0.7)
    p.add_argument("--legacy", action="store_true",
                   help="train/validate on the archived stage-1 dataset (faults6_stratified_1100, 6 classes, "
                        "880/110/110 split of the archived MaxViT baseline) instead of dataset_v2")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--loss", choices=list(LOSSES), default="dice_focal",
                   help="dicesq_focal: V-Net squared-denominator Dice + focal; bce_lovasz: weighted BCE + Lovasz hinge")
    p.add_argument("--select", choices=["loss", "iou"], default="loss", help="checkpoint selection criterion on val")
    p.add_argument("--resume", action="store_true",
                   help="continue runs/maxvit3d_<tag> from its last.pt (model/optimizer/scaler/best/stale)")
    a = p.parse_args()
    model_name, mkw, use_profile, use_trace = VARIANTS[a.variant]
    use_d2s = model_name == "maxvit3d_d2s"
    seg_loss = LOSSES[a.loss]

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local); device = torch.device(f"cuda:{local}")
    torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
    torch.backends.cudnn.benchmark = True
    global_batch = world * a.per_rank * a.accum

    out = Path(f"runs/maxvit3d_{a.tag}"); out.mkdir(parents=True, exist_ok=True)
    model = build(model_name, **mkw).to(device)
    if use_d2s and a.d2s_init > 0:
        # from scratch the branch must carry signal from step 1; atanh so that
        # tanh(param) == d2s_init.  (Zero init is for warm starts only.)
        v = float(np.arctanh(a.d2s_init))
        for m in (model.fusion32, model.fusion64):
            with torch.no_grad():
                m.context_strength.fill_(v); m.gate_strength.fill_(v)
    n_par = sum(t.numel() for t in model.parameters())
    ddp = DDP(model, device_ids=[local])

    if a.thebe:
        assert not (use_profile or use_trace), "geometry losses need the v2 geometry cache"
        tr = ThebeCubes("train", train=True, seed=a.seed) if a.thebe_fixed else \
            ThebeRandomCubes(n_per_epoch=a.thebe_n, seed=a.seed, keep_empty=a.thebe_keep_empty)
        va = ThebeCubes("val", train=False)
        cats = THEBE_CATEGORIES
    elif a.wu:
        assert not (use_profile or use_trace), "geometry losses need the v2 geometry cache"
        tr = WuVolumes("train", train=True, seed=a.seed)
        va = WuVolumes("val", train=False)
        cats = WU_CATEGORIES
    elif a.legacy:
        assert not (use_profile or use_trace), "geometry losses need the v2 geometry cache"
        tr = LegacyVolumes("train", train=True, seed=a.seed)
        va = LegacyVolumes("val", train=False)
        cats = LEGACY_CATEGORIES
    else:
        tr = GeometryVolumes(a.data, a.cache, seed=a.seed, geometry=(use_profile or use_trace))
        va = OldAugVolumes(FaultVolumes(a.data, "val"), train=False)
        cats = CATEGORIES
    assert len(tr) % global_batch == 0
    va_sub = Subset(va, list(range(rank, len(va), world)))
    vl = DataLoader(va_sub, batch_size=1, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(ddp.parameters(), lr=a.warmup_start_lr)     # wd 0.01, one group
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best, stale = 1, float("inf"), 0
    best_iou_saved, best_loss_saved = -float("inf"), float("inf")
    lr_hold, lr_stale = a.lr, 0
    if a.resume:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        ddp.module.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best, stale = ck["epoch"] + 1, ck["best"], ck["stale"]
        lr_hold, lr_stale = ck.get("lr_hold", a.lr), ck.get("lr_stale", 0)
        best_iou_saved = ck.get("best_iou_saved", -float("inf"))
        best_loss_saved = ck.get("best_loss_saved", float("inf"))
        if rank == 0: print(f"resumed from epoch {ck['epoch']} (best val loss {best:.5f}, stale {stale})", flush=True)
        del ck
    mkw_list = [f"{k}={int(v) if isinstance(v, bool) else (','.join(map(str, v)) if isinstance(v, tuple) else v)}" for k, v in mkw.items()]
    if rank == 0 and not a.resume:
        (out / "args.json").write_text(json.dumps({
            **vars(a), "model": model_name, "mkw": mkw_list, "world": world, "global_batch": global_batch,
            "recipe": "archived stage-1 recipe on DDP (same loss path / no clip / val-loss selection)",
            "data_source_sha256": (tr.source_sha256 if (a.legacy or a.wu or a.thebe) else tr.base.source_sha256), "params": n_par,
            "legacy": a.legacy, "categories": list(cats),
            "d2s": use_d2s, "profile": use_profile, "trace": use_trace}, indent=2))
        print(f"{a.variant}: {model_name} {n_par/1e6:.2f}M  world={world} global_batch={global_batch} "
              f"d2s={use_d2s} profile={use_profile} trace={use_trace}", flush=True)
    if rank == 0:
        log = open(out / "log.csv", "a" if a.resume else "w", newline=""); wr = csv.writer(log)
        if not a.resume:
            wr.writerow(["epoch", "train_loss", "train_seg", "train_profile", "train_trace", "ramp",
                         "val_loss", "val_iou", "val_dice", "val_precision", "val_recall", "lr", "seconds"]
                        + [f"val_iou_{c}" for c in cats])

    @torch.no_grad()
    def validate():
        # every rank validates its own val subset with the SAME weights: BN running
        # stats can differ per rank after the last step, so broadcast from rank 0
        for b in ddp.module.buffers():
            dist.broadcast(b, 0)
        ddp.eval(); s = torch.zeros(2 + 3 * (1 + len(cats)), dtype=torch.float64, device=device)
        for batch in vl:
            x, y, c = batch[0], batch[1], batch[2]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = ddp.module(x)
            with torch.autocast("cuda", enabled=False):
                loss = seg_loss(logits.float(), y)
            pred = torch.sigmoid(logits.float()) > .5; yb = y > .5
            s[0] += loss.item(); s[1] += 1
            tp = (pred & yb).sum(); fp = (pred & ~yb).sum(); fn = (~pred & yb).sum()
            k = int(c[0]); s[2:5] += torch.stack([tp, fp, fn]).double()
            s[5 + 3 * k: 8 + 3 * k] += torch.stack([tp, fp, fn]).double()
        dist.all_reduce(s)
        def sc(v):
            tp, fp, fn = v.tolist(); return tp / max(tp + fp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        iou, dice, P, R = sc(s[2:5])
        return dict(loss=float(s[0] / s[1]), iou=iou, dice=dice, precision=P, recall=R,
                    per_cat=[sc(s[5 + 3 * k: 8 + 3 * k])[0] for k in range(len(cats))])

    for epoch in range(start_epoch, a.epochs + 1):
        if a.sched == "cosine":
            lr_now = warmup_cosine_lr(epoch, a.epochs, a.warmup_epochs, a.warmup_start_lr, a.lr, a.min_lr)
        else:                                                   # plateau: same warmup, then held / halved lr
            lr_now = warmup_cosine_lr(epoch, a.epochs, a.warmup_epochs, a.warmup_start_lr, a.lr, a.min_lr) \
                if epoch <= a.warmup_epochs else lr_hold
        for g in opt.param_groups: g["lr"] = lr_now
        ramp = float(np.clip((epoch - a.warmup_epochs) / max(a.ramp_epochs, 1), 0.0, 1.0))
        tr.epoch = epoch
        order = np.random.default_rng(a.seed + epoch).permutation(len(tr)).tolist()
        if a.smoke: order = order[:global_batch * 3]
        sub = Subset(tr, order)
        sampler = DistributedSampler(sub, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        dl = DataLoader(sub, sampler=sampler, batch_size=a.per_rank, num_workers=a.workers, pin_memory=True)
        ddp.train(); t0 = time.time(); sums = np.zeros(4)
        for step, (x, y, _, q, nrm, valid, win) in enumerate(dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if step % a.accum == 0:
                opt.zero_grad(set_to_none=True)
            last_micro = (step + 1) % a.accum == 0 or step == len(dl) - 1
            # DDP only needs to all-reduce on the last micro-step of an accumulation group
            with (contextlib.nullcontext() if last_micro else ddp.no_sync()):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = ddp(x)
                if a.fp32_loss:
                    with torch.autocast("cuda", enabled=False):
                        seg = seg_loss(logits.float(), y)           # explicit fp32 (new protocol, both arms)
                else:
                    with torch.autocast("cuda", dtype=torch.float16):
                        seg = seg_loss(logits, y)                   # the old entry's path
                pr = tr_ = logits.float().sum() * 0
                if (use_profile or use_trace) and ramp > 0:
                    with torch.autocast("cuda", enabled=False):
                        prob = logits.float().sigmoid()
                        if use_profile: pr = profile_loss(prob, y, q.to(device), nrm.to(device), valid.to(device))
                        if use_trace:   tr_ = trace_loss(prob, y, win)
                loss = seg + ramp * (a.lambda_profile * pr + a.lambda_trace * tr_)
                scaler.scale(loss / a.accum).backward()              # no clipping, as before
            if last_micro:
                scaler.step(opt); scaler.update()
            sums += [loss.item(), seg.item(), pr.item(), tr_.item()]
            if rank == 0 and a.smoke:
                print(f"smoke step {step} loss {loss.item():.4f} seg {seg.item():.4f} pr {pr.item():.4f} tr {tr_.item():.4f}", flush=True)
        m = torch.tensor(sums / max(len(dl), 1), device=device); dist.all_reduce(m); m /= world
        v = validate(); dt = time.time() - t0
        if rank == 0:
            wr.writerow([epoch, f"{m[0]:.5f}", f"{m[1]:.5f}", f"{m[2]:.5f}", f"{m[3]:.5f}", f"{ramp:.2f}",
                         f"{v['loss']:.5f}", f"{v['iou']:.4f}", f"{v['dice']:.4f}", f"{v['precision']:.4f}",
                         f"{v['recall']:.4f}", f"{lr_now:.2e}", f"{dt:.0f}"] + [f"{c:.4f}" for c in v["per_cat"]]); log.flush()
            print(f"epoch={epoch:03d} train={m[0]:.5f} (seg {m[1]:.5f} pr {m[2]:.4f} tr {m[3]:.4f} ramp {ramp:.2f}) "
                  f"val={v['loss']:.5f} IoU={v['iou']:.4f} P={v['precision']:.3f} R={v['recall']:.3f} lr={lr_now:.2e} {dt:.0f}s", flush=True)
        score = v["loss"] if a.select == "loss" else -v["iou"]
        improved = score < best
        if improved: best, stale, lr_stale = score, 0, 0
        else:
            stale += 1; lr_stale += 1
            if a.sched == "plateau" and epoch > a.warmup_epochs and lr_stale >= a.lr_patience:
                lr_hold, lr_stale = max(lr_hold * a.lr_factor, a.min_lr), 0
                if rank == 0: print(f"   plateau: lr -> {lr_hold:.2e}", flush=True)
        if rank == 0:
            if getattr(ddp.module, "use_dyn", False) and ddp.module.dyn64.last_offset_stats:
                st = ddp.module.dyn64.last_offset_stats
                print(f"   dyn64 offset: mean|d| {st['mean_abs_coarse_vox']:.3f} coarse-vox, "
                      f"saturated {100*st['saturated_frac']:.1f}%, out-of-range {100*st['out_of_range_frac']:.2f}%", flush=True)
                with open(out / "offset_stats.jsonl", "a") as fh: fh.write(json.dumps({"epoch": epoch, **st}) + "\n")
            state = {"model": ddp.module.state_dict(), "epoch": epoch, "val_loss": v["loss"], "val": v,
                     "args": {**vars(a), "model": model_name, "mkw": mkw_list}}
            if improved: torch.save(state, out / "best.pt")
            # Explicit companions; best.pt retains the configured historical selection.
            # Resuming an older run starts companion tracking at resume, not retroactively.
            if v["iou"] > best_iou_saved:
                best_iou_saved = v["iou"]
                torch.save({**state, "selection": "val_iou"}, out / "best_iou.pt")
            if v["loss"] < best_loss_saved:
                best_loss_saved = v["loss"]
                torch.save({**state, "selection": "val_loss"}, out / "best_loss.pt")
            torch.save({"model": ddp.module.state_dict(), "epoch": epoch, "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "best": best, "stale": stale,
                        "best_iou_saved": best_iou_saved, "best_loss_saved": best_loss_saved,
                        "lr_hold": lr_hold, "lr_stale": lr_stale}, out / "last.pt")
        stop = torch.tensor(int(stale >= a.patience or a.smoke), device=device); dist.broadcast(stop, 0)
        if stop.item():
            if rank == 0: print(f"stop at epoch {epoch}; best val loss {best:.5f}", flush=True)
            break
    if rank == 0: log.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
