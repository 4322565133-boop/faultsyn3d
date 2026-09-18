"""3D MaxViT-tiny encoder (timm_3d) + a U-Net-style 3D conv decoder.

Mirrors FAULTSEG3D/models/maxvitunet.py's encoder/decoder pattern, ported to
3D: Conv3d/InstanceNorm3d, trilinear upsampling, and timm_3d's 3D port of
maxvit_tiny_tf_224 (features_only, 5 stages at strides 2/4/8/16/32) instead
of timm's 2D version. Trained from scratch (random init): timm_3d's 2D->3D
weight inflation for maxvit_tiny_tf_224's relative-position-bias table is
currently broken (AssertionError in resize_rel_pos_bias_table), so
pretrained=True is not usable here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm_3d


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MaxViT3DUNet(nn.Module):
    """Three switchable fixes, each addressing a measured defect of the stock model.

    full_res_skip   The finest backbone feature is stride 2, so the stock decoder
                    predicts a 1.5-voxel-thick label from a 64^3 map upsampled by
                    interpolation.  87 % of its false positives sat 1-2 voxels from
                    the label: it draws the fault too thick.  This adds a shallow
                    full-resolution encoder branch and one more decoder level.
    drop_path_rate  The stock model has no stochastic depth at all; trained from
                    scratch on 800 volumes it overfit (train/val loss gap +0.18).
    img_size        timm_3d derives the attention partition from img_size/32; the
                    224 default collapses to a (2,2,2) window in 3D, i.e. attention
                    over 8 tokens.  128 gives (4,4,4).
    """
    def __init__(self, n_channels=1, n_classes=1, backbone_name="maxvit_tiny_tf_224.in1k",
                 pretrained=False, full_res_skip=False, drop_path_rate=0.0, img_size=None,
                 stem_ch=16):
        super().__init__()
        kw = dict(drop_path_rate=drop_path_rate)
        if img_size is not None:
            kw["img_size"] = img_size
        self.backbone = timm_3d.create_model(
            backbone_name, pretrained=pretrained, in_chans=n_channels, features_only=True, **kw,
        )
        self.full_res_skip = full_res_skip
        if full_res_skip:
            self.enc_full = DoubleConv3D(n_channels, stem_ch)
        self.chs = list(self.backbone.feature_info.channels())  # [64, 64, 128, 256, 512] @ stride 2/4/8/16/32
        assert len(self.chs) == 5, f"expected 5 feature scales, got {len(self.chs)}"

        c0, c1, c2, c3, c4 = self.chs
        self.dec4 = DoubleConv3D(c4 + c3, c3)
        self.dec3 = DoubleConv3D(c3 + c2, c2)
        self.dec2 = DoubleConv3D(c2 + c1, c1)
        self.dec1 = DoubleConv3D(c1 + c0, c0)
        if full_res_skip:
            self.dec0 = DoubleConv3D(c0 + stem_ch, 32)
        else:
            self.dec0 = DoubleConv3D(c0, 32)
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.backbone(x)  # strides 2,4,8,16,32

        d = F.interpolate(f4, size=f3.shape[2:], mode="trilinear", align_corners=False)
        d = self.dec4(torch.cat([d, f3], dim=1))

        d = F.interpolate(d, size=f2.shape[2:], mode="trilinear", align_corners=False)
        d = self.dec3(torch.cat([d, f2], dim=1))

        d = F.interpolate(d, size=f1.shape[2:], mode="trilinear", align_corners=False)
        d = self.dec2(torch.cat([d, f1], dim=1))

        d = F.interpolate(d, size=f0.shape[2:], mode="trilinear", align_corners=False)
        d = self.dec1(torch.cat([d, f0], dim=1))

        if self.full_res_skip:
            d = F.interpolate(d, size=x.shape[2:], mode="trilinear", align_corners=False)
            d = self.dec0(torch.cat([d, self.enc_full(x)], dim=1))
        else:
            d = self.dec0(d)
            d = F.interpolate(d, size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.outc(d)
