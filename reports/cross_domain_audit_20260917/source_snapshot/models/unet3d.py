"""Plain 3D U-Net baseline (three pooling levels, InstanceNorm).

Copied verbatim from the archived stage-1 project's train_3dunet.py so the
baseline is the same network that project reported on.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout), nn.ReLU(inplace=True),
        )


class UNet3D(nn.Module):
    def __init__(self, base=16, out_channels=1, checkpoint_highres=False):
        super().__init__()
        # optional gradient checkpointing of the two 128^3 blocks (InstanceNorm only, so recomputation is
        # exact); off by default so the baseline is byte-identical to the archived network
        self.checkpoint_highres = bool(checkpoint_highres)
        self.enc1, self.enc2, self.enc3 = DoubleConv(1, base), DoubleConv(base, base * 2), DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.mid = DoubleConv(base * 4, base * 8)
        self.up3, self.dec3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2), DoubleConv(base * 8, base * 4)
        self.up2, self.dec2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2), DoubleConv(base * 4, base * 2)
        self.up1, self.dec1 = nn.ConvTranspose3d(base * 2, base, 2, 2), DoubleConv(base * 2, base)
        self.out = nn.Conv3d(base, out_channels, 1)

    def _hr(self, block, x):
        if self.training and self.checkpoint_highres:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        a = self._hr(self.enc1, x); b = self.enc2(self.pool(a)); c = self.enc3(self.pool(b)); x = self.mid(self.pool(c))
        x = self.dec3(torch.cat((self.up3(x), c), 1)); x = self.dec2(torch.cat((self.up2(x), b), 1))
        return self.out(self._hr(self.dec1, torch.cat((self.up1(x), a), 1)))


