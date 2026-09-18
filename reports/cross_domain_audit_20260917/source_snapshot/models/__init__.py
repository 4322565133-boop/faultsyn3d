"""Three segmentation baselines sharing one interface: (B,1,D,H,W) -> logits."""

from .unet3d import UNet3D
from .resnet3d import ResNet3DUNet
from .maxvit3d import MaxViT3DUNet
from .maxvit3d_enhanced import EnhancedMaxViT3D
from .dbvit3d import DoubleBlockViTUNet
from .compact_maxvit3d import CompactMaxViT3D
from .unet_grid3d import UNetGrid3D
from .unet_maxvit3d import UNetMaxViT3D


def build(name: str, **kw):
    name = name.lower()
    if name == "unet_grid3d":
        return UNetGrid3D(**kw)
    if name == "unet_maxvit3d":
        # UNet-L + MaxViT-style attention at enc3 / mid (models/unet_maxvit3d.py)
        kw = dict(kw)
        for k in ("attn32", "attn16"):
            if isinstance(kw.get(k), str):
                kw[k] = tuple(t for t in kw[k].split(",") if t)
        return UNetMaxViT3D(**kw)
    if name == "compact_maxvit3d":
        return CompactMaxViT3D(**kw)
    if name == "unet3d":
        return UNet3D(base=kw.get("base", 16), checkpoint_highres=bool(kw.get("checkpoint_highres", False)))
    if name == "resnet3d":
        return ResNet3DUNet(pretrained=False)
    if name == "maxvit3d":
        return MaxViT3DUNet(pretrained=False, **{k: v for k, v in kw.items() if k != "base"})
    if name == "dbvit":
        # DoubleBlock-ViT U-Net (Nguyen-Tat et al. 2026), faithful port of the official code
        return DoubleBlockViTUNet(**{k: v for k, v in kw.items() if k in ("n_channels", "blocks", "full_res_skip")})
    if name == "maxvit3d_dyn":
        # all3 + shallow-guided dynamic upsampling at 32^3 -> 64^3 (models/dyn_upsample3d.py)
        return EnhancedMaxViT3D(dyn_up=True, pretrained=False,
                                **{k: v for k, v in kw.items() if k not in ("base", "d2s", "dyn_up")})
    if name == "maxvit3d_d2s":
        # all3 backbone + D2S fusion at the 32^3 / 64^3 skips (models/d2s_fusion3d.py)
        return EnhancedMaxViT3D(d2s=True, pretrained=False,
                                **{k: v for k, v in kw.items() if k not in ("base", "d2s")})
    raise ValueError(name)
