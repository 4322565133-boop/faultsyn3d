"""DoubleBlock-ViT U-Net (Nguyen-Tat et al., CMPB 274 (2026) 109165), ported for
single-channel seismic volumes with a single fault logit.

Source of truth is the authors' code, models/DB_MaxViT.py in
https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment -- ported block by
block.  Where the code and the paper differ, the CODE is followed:

  * MBConv has NO residual connection (it is a plain nn.Sequential); the two
    attention/FFN pairs have pre-norm residuals.
  * Each MaxViT block = PE-MBConv -> [window attn + FFN] x 2.  No grid attention.
  * Window (4,4,4), dim_head 16, attention/FFN dropout 0.1, FFN mult 3.
  * Encoder stage = strided 3x3x3 conv (stride 2) + InstanceNorm, then
    `blocks[i]` MaxViT blocks.  Default blocks (1, 1, 2) is the BraTS2021 config.
  * Bottleneck = strided conv + [1x1x1, IN, GELU, depthwise 3x3x3, IN, GELU].
  * DPF decoder (Decoder1) for the three deep skips; the last stage (Up2)
    upsamples 64^3 -> 128^3 with plain convs and -- in the official code --
    IGNORES the full-resolution stem feature it is handed.  `full_res_skip=True`
    concatenates it instead (our variant, off by default).
  * Loss in the official repo: 0.75 * soft Dice (squared denominator) + 0.25 * BCE.

With n_channels=16 the authors report 7.8 M parameters (4-channel input,
3-class output); the 1-in / 1-out port is marginally smaller.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum


class ProjectExciteLayer(nn.Module):
    """Project & Excite (Rickmann et al. 2019): axis-wise average projections,
    summed back to 3-D, then a 1x1x1 bottleneck -> sigmoid -> spatial gate."""
    def __init__(self, num_channels, reduction_ratio=4):
        super().__init__()
        r = num_channels // reduction_ratio
        self.conv_c = nn.Conv3d(num_channels, r, 1)
        self.conv_cT = nn.Conv3d(r, num_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, D, H, W = x.shape
        sw = F.adaptive_avg_pool3d(x, (1, 1, W))
        sh = F.adaptive_avg_pool3d(x, (1, H, 1))
        sd = F.adaptive_avg_pool3d(x, (D, 1, 1))
        s = sw.view(b, c, 1, 1, W) + sh.view(b, c, 1, H, 1) + sd.view(b, c, D, 1, 1)
        return x * torch.sigmoid(self.conv_cT(self.act(self.conv_c(s))))


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class Downsampling(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=2, padding=1)
        self.norm = nn.InstanceNorm3d(cout, affine=True)

    def forward(self, x):
        return self.norm(self.conv(x))


def MBConv(dim_in, dim_out, expansion_rate=2):
    """PE-MBConv as in the official code: no residual around it."""
    h = int(expansion_rate * dim_out)
    return nn.Sequential(
        nn.Conv3d(dim_in, h, 1), nn.InstanceNorm3d(h, affine=True), nn.GELU(),
        nn.Conv3d(h, h, 3, padding=1, groups=h), nn.InstanceNorm3d(h, affine=True), nn.GELU(),
        ProjectExciteLayer(h),
        nn.Conv3d(h, dim_out, 1), nn.InstanceNorm3d(dim_out, affine=True))


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=3, dropout=0.):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Window (block) attention with a learned 3-D relative position bias."""
    def __init__(self, dim, dim_head=16, dropout=0., window_size=(4, 4, 4)):
        super().__init__()
        assert dim % dim_head == 0
        self.heads = dim // dim_head; self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        w1, w2, w3 = window_size
        self.rel_pos_bias = nn.Embedding((2 * w1 - 1) * (2 * w2 - 1) * (2 * w3 - 1), self.heads)
        grid = torch.stack(torch.meshgrid(torch.arange(w1), torch.arange(w2), torch.arange(w3), indexing="ij"))
        grid = rearrange(grid, "c i j k -> (i j k) c")
        rel = rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")
        rel[..., 0] += w1 - 1; rel[..., 1] += w2 - 1; rel[..., 2] += w3 - 1
        idx = (rel * torch.tensor([(2 * w2 - 1) * (2 * w3 - 1), (2 * w3 - 1), 1])).sum(-1)
        self.register_buffer("rel_pos_indices", idx, persistent=False)

    def forward(self, x):
        b, X, Y, Z, w1, w2, w3, _ = x.shape
        x = rearrange(x, "b x y z w1 w2 w3 d -> (b x y z) (w1 w2 w3) d")
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q * self.scale, k)
        sim = sim + rearrange(self.rel_pos_bias(self.rel_pos_indices), "i j h -> h i j")
        out = einsum("b h i j, b h j d -> b h i d", self.attend(sim), v)
        out = rearrange(out, "b h (w1 w2 w3) d -> b w1 w2 w3 (h d)", w1=w1, w2=w2, w3=w3)
        out = self.to_out(out)
        return rearrange(out, "(b x y z) ... -> b x y z ...", x=X, y=Y, z=Z)


class MaxViTBlock(nn.Module):
    """PE-MBConv -> [window attention + FFN] x 2   (the "double block")."""
    def __init__(self, dim, dim_head=16, window_size=(4, 4, 4), dropout=0.1):
        super().__init__()
        w1, w2, w3 = window_size
        part = Rearrange("b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d", w1=w1, w2=w2, w3=w3)
        unpart = Rearrange("b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)")
        self.net = nn.Sequential(
            MBConv(dim, dim),
            part,
            PreNormResidual(dim, Attention(dim, dim_head, dropout, window_size)),
            PreNormResidual(dim, FeedForward(dim, dropout=dropout)),
            unpart, part,
            PreNormResidual(dim, Attention(dim, dim_head, dropout, window_size)),
            PreNormResidual(dim, FeedForward(dim, dropout=dropout)),
            unpart)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, cin, cout, n_blocks, window_size=(4, 4, 4)):
        super().__init__()
        self.down = Downsampling(cin, cout)
        # the official code re-applies ONE MaxViT_Block instance n times (shared
        # weights); replicated here as-is for fidelity
        self.block = MaxViTBlock(cout, dim_head=16, window_size=window_size)
        self.n = n_blocks

    def forward(self, x):
        x = self.down(x)
        for _ in range(self.n):
            x = self.block(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.down = Downsampling(cin, cout)
        self.net = nn.Sequential(nn.Conv3d(cout, cout, 1), nn.InstanceNorm3d(cout, affine=True), nn.GELU(),
                                 nn.Conv3d(cout, cout, 3, padding=1, groups=cout), nn.InstanceNorm3d(cout, affine=True), nn.GELU())

    def forward(self, x):
        return self.net(self.down(x))


class DPFDecoder(nn.Module):
    """Dual-Path Fusion skip (Decoder1 in the official code), verbatim."""
    def __init__(self, cin, cout):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = nn.Conv3d(cin, cout, 1, bias=False)
        self.norm = nn.InstanceNorm3d(cout, affine=True)
        self.conv_block = nn.Conv3d(2 * cout, cout, 1, bias=False)
        self.point_wise_conv = nn.Sequential(nn.Conv3d(cout, cout, 1, bias=False), nn.InstanceNorm3d(cout, affine=True))
        self.depth_wise_conv = nn.Sequential(nn.Conv3d(cout, cout, 3, padding=1, groups=cout),
                                             nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01))
        self.conv_2 = nn.Conv3d(1, 2, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.PE = ProjectExciteLayer(cout)

    def forward(self, x1, skip):
        up1 = self.norm(self.conv(self.upsample(x1)))                 # X~1
        a2 = torch.amax(self.depth_wise_conv(skip), dim=1, keepdim=True)   # Att2 from DWC(X2)
        a1 = torch.amax(up1, dim=1, keepdim=True)                     # Att1
        att = F.softmax(self.conv_2(a1 + a2), dim=1)                  # Out1 -> softmax -> split
        g1, g2 = att.split(1, dim=1)
        C = skip.size(1)
        x2 = g2.repeat(1, C, 1, 1, 1) * skip + skip                   # X'2
        x1 = g1.repeat(1, C, 1, 1, 1) * up1 + up1                     # X'1
        z1 = x1 * torch.sigmoid(x2); z2 = x2 * torch.sigmoid(x1)      # cross gating
        z = torch.sigmoid(self.point_wise_conv(z1 + z2))
        z = skip * z
        z = torch.sigmoid(self.PE(self.point_wise_conv(z)))
        z = up1 * z
        return self.leaky(self.conv_block(torch.cat([z, skip], 1)))


class LastUp(nn.Module):
    """Up2 in the official code.  As written there it does NOT use the stem skip;
    `use_skip=True` concatenates it (our full-resolution variant)."""
    def __init__(self, cin, cout, use_skip=False, skip_ch=0):
        super().__init__()
        self.use_skip = use_skip
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = nn.Conv3d(cin, cin, 1, bias=False)
        c_in2 = cin + (skip_ch if use_skip else 0)
        self.convLK_in = nn.Sequential(nn.InstanceNorm3d(c_in2, affine=True), nn.LeakyReLU(0.01),
                                       nn.Conv3d(c_in2, 2 * cout, 3, padding=1, bias=False))
        self.convLK_out = nn.Sequential(nn.InstanceNorm3d(2 * cout, affine=True), nn.LeakyReLU(0.01),
                                        nn.Conv3d(2 * cout, cout, 3, padding=1, bias=False))

    def forward(self, x1, skip):
        x = self.conv(self.upsample(x1))
        if self.use_skip:
            x = torch.cat([x, skip], 1)
        return self.convLK_out(self.convLK_in(x))


class DoubleBlockViTUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, n_channels=16, blocks=(1, 1, 2),
                 window_size=(4, 4, 4), full_res_skip=False):
        super().__init__()
        c = n_channels
        self.conv = DoubleConv(in_channels, 2 * c)                     # 128^3, 32 ch
        self.enc1 = Encoder(2 * c, 4 * c, blocks[0], window_size)       # 64^3,  64 ch
        self.enc2 = Encoder(4 * c, 8 * c, blocks[1], window_size)       # 32^3, 128 ch
        self.enc3 = Encoder(8 * c, 16 * c, blocks[2], window_size)      # 16^3, 256 ch
        self.bottleneck = Bottleneck(16 * c, 32 * c)                    #  8^3, 512 ch
        self.dec1 = DPFDecoder(32 * c, 16 * c)                          # 16^3
        self.dec2 = DPFDecoder(16 * c, 8 * c)                           # 32^3
        self.dec3 = DPFDecoder(8 * c, 4 * c)                            # 64^3
        self.dec4 = LastUp(4 * c, c, use_skip=full_res_skip, skip_ch=2 * c)   # 128^3
        self.out = nn.Conv3d(c, n_classes, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        s0 = self.conv(x)
        s1 = self.enc1(s0); s2 = self.enc2(s1); s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        d = self.dec1(b, s3); d = self.dec2(d, s2); d = self.dec3(d, s1)
        d = self.dec4(d, s0)
        return self.out(d)
