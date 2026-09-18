"""U-Net-L preserved, with switchable grid-only residual attention.

16^3 bottleneck: test whether distant context helps missing curved/branch faults.
32^3 decoder: condition reconstructed detail on context without gating away skips.
These are hypotheses, not guarantees about precision or topology.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm_3d.models.maxxvit import PartitionAttentionCl, MaxxVitTransformerCfg
from .unet3d import UNet3D


class GridContext3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        cfg = MaxxVitTransformerCfg(dim_head=24, expand_ratio=2,
                                   window_size=(4, 4, 4), grid_size=(4, 4, 4),
                                   attn_drop=0., proj_drop=0., init_values=None)
        self.grid = PartitionAttentionCl(channels, partition_type='grid', cfg=cfg, drop_path=0.)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, x):
        if any(s % 4 for s in x.shape[2:]):
            raise ValueError('Grid context spatial sizes must be divisible by four')
        return self.grid(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class UNetGrid3D(UNet3D):
    def __init__(self, base=42, out_channels=1, grid_mid=True, grid_dec=True, checkpoint_highres=True):
        super().__init__(base=base, out_channels=out_channels)
        self.checkpoint_highres = bool(checkpoint_highres)
        # Extra modules must not change the RNG stream of the shared U-Net or
        # subsequent training. Same seed -> exactly identical common weights.
        with torch.random.fork_rng(devices=[]):
            self.grid_mid = GridContext3D(base * 8) if grid_mid else nn.Identity()
            self.grid_dec = GridContext3D(base * 4) if grid_dec else nn.Identity()

    def detail(self, block, x):
        if self.training and self.checkpoint_highres:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        a = self.detail(self.enc1, x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        x = self.grid_mid(self.mid(self.pool(c)))
        x = self.grid_dec(self.dec3(torch.cat((self.up3(x), c), 1)))
        x = self.dec2(torch.cat((self.up2(x), b), 1))
        return self.out(self.detail(self.dec1, torch.cat((self.up1(x), a), 1)))
