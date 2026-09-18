"""2-D MaxViT encoder (shared across slices) + lightweight cross-slice adapters + 3-D U-Net decoder.

Design brief (2026-09-18): keep the validated 2-D MaxViT backbone (timm `maxvit_tiny_tf_224`, ImageNet
weights, the same four scales the 2-D MaxViTUNet used) for in-plane texture / discontinuity / block +
grid attention, and let small residual adapters learn only the third axis (slice-to-slice continuity of
fault surfaces).  Nothing else in the backbone is 3-D.

    x (B,1,D,H,W)  ->  W vertical sections (B*W,1,D,H)  ->  2-D MaxViT features at strides 2/4/8/16
      -> restack to (B,C,D/r,H/r,W) -> average over r adjacent slices -> isotropic (B,C,D/r,H/r,W/r)
      -> DepthAdapter (residual, zero-init: 1x1x1 down -> depthwise (1,1,k) conv along W -> GN+GELU -> 1x1x1 up)
      -> 3-D U-Net decoder with a full-resolution conv skip (conv-IN-ReLU DoubleConv, as UNet3D)  ->  logits

The slice axis is the last one (W); Thebe training rotates the horizontal plane, so inline and crossline
sections are both seen.  Averaging over r slices makes the deep skips isotropic, so the decoder is the
plain 3-D U-Net decoder (0.26 TFLOPs) instead of an anisotropic one (0.52 TFLOPs); the adapters see
r-slice-averaged sections at depth and the full slice resolution at the stride-2 level and the input.
"""
from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .unet3d import DoubleConv


class DepthAdapter(nn.Module):
    """Residual bottleneck that mixes information only along the slice axis (last dim)."""

    def __init__(self, ch, ratio=4, k=3):
        super().__init__()
        h = max(ch // ratio, 8)
        self.down = nn.Conv3d(ch, h, 1)
        self.dw = nn.Conv3d(h, h, (1, 1, k), padding=(0, 0, k // 2), groups=h)
        self.norm = nn.GroupNorm(1, h)
        self.act = nn.GELU()
        self.up = nn.Conv3d(h, ch, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)      # starts as identity

    def forward(self, x):
        return x + self.up(self.act(self.norm(self.dw(self.down(x)))))


class MaxViT2DAdapter3D(nn.Module):
    def __init__(self, backbone="maxvit_tiny_tf_224", pretrained=True, levels=4, adapter=True, adapter_levels="all",
                 adapter_ratio=4, adapter_k=3, in_slices=1, dec=(128, 64, 32, 16), full_ch=16, size=128,
                 checkpoint_backbone=False, out_channels=1):
        super().__init__()
        assert in_slices in (1, 3)
        self.in_slices = in_slices
        self.backbone = timm.create_model(backbone, pretrained=bool(pretrained), in_chans=in_slices, features_only=True,
                                          out_indices=tuple(range(levels)), img_size=size)
        if checkpoint_backbone:
            self.backbone.set_grad_checkpointing(True)
        self.chs = list(self.backbone.feature_info.channels())
        self.red = list(self.backbone.feature_info.reduction())
        if adapter_levels == "all":
            use = set(range(levels))
        elif adapter_levels == "deep":
            use = {levels - 1}
        else:
            use = {int(t) for t in str(adapter_levels).split(",") if t}
        self.adapters = nn.ModuleList([DepthAdapter(c, adapter_ratio, adapter_k) if (adapter and i in use) else nn.Identity()
                                       for i, c in enumerate(self.chs)])
        # decoder widths `dec` are ordered deep -> shallow: dec[0] at the second-deepest skip, dec[-1] at full resolution
        dec = list(dec)[-levels:]
        self.enc_full = DoubleConv(1, full_ch)
        blocks, prev = [], self.chs[-1]
        for j, i in enumerate(range(levels - 2, -1, -1)):
            blocks.append(DoubleConv(prev + self.chs[i], dec[j])); prev = dec[j]
        self.dec = nn.ModuleList(blocks)
        self.dec_full = DoubleConv(prev + full_ch, dec[-1])
        self.out = nn.Conv3d(dec[-1], out_channels, 1)

    def _slices(self, x):
        B, _, D, H, W = x.shape
        if self.in_slices == 3:                                   # neighbours as channels (edge replicate)
            xp = F.pad(x, (1, 1), mode="replicate")
            x = torch.cat((xp[..., :-2], xp[..., 1:-1], xp[..., 2:]), 1)
        return x.permute(0, 4, 1, 2, 3).reshape(B * W, x.shape[1], D, H)

    def forward(self, x):
        B, _, D, H, W = x.shape
        feats = self.backbone(self._slices(x))
        skips = []
        for f, r, ad in zip(feats, self.red, self.adapters):
            f = f.reshape(B, W, f.shape[1], f.shape[2], f.shape[3]).permute(0, 2, 3, 4, 1)      # (B,C,D/r,H/r,W)
            f = F.avg_pool3d(f, (1, 1, r)) if r > 1 else f                                       # isotropic
            skips.append(ad(f))
        y = skips[-1]
        for blk, skip in zip(self.dec, skips[-2::-1]):
            y = blk(torch.cat((F.interpolate(y, scale_factor=2, mode="trilinear", align_corners=False), skip), 1))
        y = self.dec_full(torch.cat((F.interpolate(y, scale_factor=2, mode="trilinear", align_corners=False), self.enc_full(x)), 1))
        return self.out(y)
