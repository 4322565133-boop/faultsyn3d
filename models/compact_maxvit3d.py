"""M0: three downsamplings, full-resolution detail, genuine block+grid attention.

9,489,043 parameters at base=36. This is the architecture-only starting point;
no D2S, DPF, directional module or geometric auxiliary supervision is included.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from timm_3d.models.maxxvit import MaxxVitBlock, MaxxVitConvCfg, MaxxVitTransformerCfg
from .maxvit3d import DoubleConv3D


class CompactMaxViT3D(nn.Module):
    def __init__(self, base=36, drop_path_rate=0.2, checkpoint_highres=True):
        super().__init__()
        a, b, c, d = base, base * 2, base * 4, base * 8
        if c % 16:
            raise ValueError('base * 4 must be divisible by head dimension 16')
        self.checkpoint_highres = bool(checkpoint_highres)
        cc = MaxxVitConvCfg(expand_ratio=2)
        tc = MaxxVitTransformerCfg(dim_head=16, expand_ratio=3,
                                  window_size=(4, 4, 4), grid_size=(4, 4, 4))
        self.stem = DoubleConv3D(1, a)
        self.local64 = nn.Sequential(nn.Conv3d(a, b, 3, stride=2, padding=1), DoubleConv3D(b, b))
        self.down32 = nn.Conv3d(b, c, 3, stride=2, padding=1)
        self.max32 = MaxxVitBlock(c, c, conv_cfg=cc, transformer_cfg=tc, drop_path=0.)
        self.down16 = nn.Conv3d(c, d, 3, stride=2, padding=1)
        self.max16 = nn.Sequential(*[
            MaxxVitBlock(d, d, conv_cfg=cc, transformer_cfg=tc, drop_path=drop_path_rate * f)
            for f in (0.5, 1.)])
        self.dec32 = DoubleConv3D(d + c, c)
        self.dec64 = DoubleConv3D(c + b, b)
        self.dec128 = DoubleConv3D(b + a, a)
        self.out = nn.Conv3d(a, 1, 1)
        # Standard fresh initialization; no pretrained/checkpoint weights.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _detail(self, module, x):
        # Only InstanceNorm blocks without running statistics are recomputed.
        # Recomputing MBConv BatchNorm would update its buffers twice.
        if self.training and self.checkpoint_highres:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(s % 32 for s in x.shape[2:]):
            raise ValueError('Expected B,1,D,H,W with spatial sizes divisible by 32')
        def up(t, ref):
            return F.interpolate(t, size=ref.shape[2:], mode='trilinear', align_corners=False)
        f0 = self._detail(self.stem, x)
        f1 = self.local64(f0)
        f2 = self.max32(self.down32(f1))
        f3 = self.max16(self.down16(f2))
        d = self.dec32(torch.cat([up(f3, f2), f2], 1))
        d = self.dec64(torch.cat([up(d, f1), f1], 1))
        d = self._detail(self.dec128, torch.cat([up(d, f0), f0], 1))
        return self.out(d)
