"""MaxViT baseline with optional fusion at strides 4 and 2."""
import torch
from torch.nn import functional as F
from .maxvit3d import MaxViT3DUNet
from .d2s_fusion3d import D2SFusion3D
from .dyn_upsample3d import DynamicUpsample3D


class EnhancedMaxViT3D(MaxViT3DUNet):
    """all3 backbone/decoder with two independent, switchable changes:

    d2s      deep-to-shallow gated fusion at the 32^3 / 64^3 skips (negative
             result on v2: 141 epochs from scratch, identical to baseline)
    dyn_up   replace the fixed trilinear upsample 32^3 -> 64^3 (the input to
             dec1) with shallow-guided dynamic sampling (models/dyn_upsample3d.py)
    """
    def __init__(self, d2s=False, dyn_up=False, dyn_max_offset=0.5, **kwargs):
        super().__init__(**kwargs)
        self.use_d2s, self.use_dyn = d2s, dyn_up
        c0, c1, c2, c3, c4 = self.chs
        if d2s:
            self.fusion32 = D2SFusion3D(c1, c2, [c2, c3, c4])
            self.fusion64 = D2SFusion3D(c0, c1, [c1, c2, c3])
        if dyn_up:
            # dec2 output has c1 channels at 32^3; the guide is f0 (c0 ch) at 64^3
            self.dyn64 = DynamicUpsample3D(coarse_ch=c1, fine_ch=c0, max_offset=dyn_max_offset)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.backbone(x)
        def up(t, ref):
            return F.interpolate(t, size=ref.shape[2:], mode='trilinear', align_corners=False)
        d = self.dec4(torch.cat([up(f4, f3), f3], 1))
        d = self.dec3(torch.cat([up(d, f2), f2], 1))
        d = up(d, f1)
        s = self.fusion32(f1, d, [f2, f3, f4]) if self.use_d2s else f1
        d = self.dec2(torch.cat([d, s], 1))
        d = self.dyn64(d, f0) if self.use_dyn else up(d, f0)
        s = self.fusion64(f0, d, [f1, f2, f3]) if self.use_d2s else f0
        d = self.dec1(torch.cat([d, s], 1))
        if self.full_res_skip:
            d = self.dec0(torch.cat([up(d, x), self.enc_full(x)], 1))
        else:
            d = up(self.dec0(d), x)
        return self.outc(d)
