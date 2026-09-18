"""3D ResNet-50 encoder (timm_3d) + a U-Net-style 3D conv decoder.

Same encoder/decoder pattern as maxvit3d.py (timm_3d features_only backbone,
5 stages at strides 2/4/8/16/32, DoubleConv3D decoder blocks), swapped to
resnet50 so it can be trained under the identical FaultVitNet recipe for a
controlled architecture comparison. Unlike maxvit_tiny_tf_224, resnet50's
2D->3D pretrained-weight inflation actually works in timm_3d (no relative-
position-bias table to resize) - but pretrained=False is kept as the default
here so the comparison against MaxViT3D (necessarily trained from scratch)
isn't confounded by one model getting ImageNet pretraining and the other not.
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


class ResNet3DUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, backbone_name="resnet50", pretrained=False):
        super().__init__()
        self.backbone = timm_3d.create_model(
            backbone_name, pretrained=pretrained, in_chans=n_channels, features_only=True,
        )
        self.chs = list(self.backbone.feature_info.channels())  # [64, 256, 512, 1024, 2048] @ stride 2/4/8/16/32
        assert len(self.chs) == 5, f"expected 5 feature scales, got {len(self.chs)}"

        c0, c1, c2, c3, c4 = self.chs
        self.dec4 = DoubleConv3D(c4 + c3, c3)
        self.dec3 = DoubleConv3D(c3 + c2, c2)
        self.dec2 = DoubleConv3D(c2 + c1, c1)
        self.dec1 = DoubleConv3D(c1 + c0, c0)
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

        d = self.dec0(d)
        d = F.interpolate(d, size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.outc(d)
