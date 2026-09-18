"""UNet-L with MaxViT-style attention at the two deep encoder levels.

UNet3D(base=42) is kept exactly (same modules, same init under the same seed);
attention is appended after the DoubleConv of enc3 (32^3, 168 ch) and mid
(16^3, 336 ch), so every dense 3^3 conv the U-Net has stays in place:

    enc3:  DoubleConv -> block attn (4^3 window, local) -> grid attn (4^3 grid, sparse global)
    mid:   DoubleConv -> block attn (4^3 window, local) -> global attn (all 4096 tokens)

`replace=True` instead drops the second 3^3 conv of enc3 / mid, so each deep
stage becomes conv -> attention like a MaxViT stage (MBConv -> attention).

Why here (val diagnosis of UNet-L @56, docs/UNET_GRID_PR_DIAGNOSIS_20260916.json):
99% of FP lie within 2 voxels of a label (thickness, not context), while 14% of
FN are whole missed pieces, 25% for listric - faint low-throw parts of surfaces
whose clear parts are elsewhere in the cube.  Long-range along-surface context
is the one thing the conv U-Net cannot do cheaply; the decoder is left alone.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm_3d.layers import DropPath, Mlp
from timm_3d.models.maxxvit import MaxxVitTransformerCfg, PartitionAttentionCl

from .unet3d import UNet3D


class GlobalAttention3D(nn.Module):
    """Pre-norm dense self-attention over every token of a (B,D,H,W,C) map + MLP.

    Factorised learned position embedding (one table per axis, summed) goes into
    the attention branch only.  SDPA keeps memory linear in tokens (no 4096^2
    map is stored), so this is affordable at 16^3.
    """

    def __init__(self, dim, size=(16, 16, 16), dim_head=24, mlp_ratio=2, drop_path=0.):
        super().__init__()
        assert dim % dim_head == 0
        self.heads, self.dim_head = dim // dim_head, dim_head
        self.pos = nn.ParameterList([nn.Parameter(torch.zeros(s, dim)) for s in size])
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _pos(self, D, H, W):
        def axis(p, n):                      # (s, C) -> (n, C), resampled only if the size differs
            if p.shape[0] == n:
                return p
            return F.interpolate(p.t()[None], size=n, mode="linear", align_corners=True)[0].t()
        pz, py, px = axis(self.pos[0], D), axis(self.pos[1], H), axis(self.pos[2], W)
        return pz[:, None, None] + py[None, :, None] + px[None, None, :]      # (D,H,W,C)

    def forward(self, x):
        B, D, H, W, C = x.shape
        t = x.reshape(B, D * H * W, C)
        h = self.norm1(t + self._pos(D, H, W).reshape(1, -1, C).to(t.dtype))
        q, k, v = self.qkv(h).reshape(B, -1, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)                             # (B, heads, N, dh)
        t = t + self.drop_path1(self.proj(o.transpose(1, 2).reshape(B, -1, C)))
        t = t + self.drop_path2(self.mlp(self.norm2(t)))
        return t.reshape(B, D, H, W, C)


class AttnStage(nn.Module):
    """Sequence of channels-last attention blocks applied to an NCDHW feature map."""

    def __init__(self, dim, kinds, dim_head=24, mlp_ratio=2, drop_path=0., global_size=(16, 16, 16)):
        super().__init__()
        cfg = MaxxVitTransformerCfg(dim_head=dim_head, expand_ratio=mlp_ratio,
                                    window_size=(4, 4, 4), grid_size=(4, 4, 4))
        blocks = []
        for kind in kinds:
            if kind in ("block", "grid"):
                blocks.append(PartitionAttentionCl(dim, partition_type=kind, cfg=cfg, drop_path=drop_path))
            elif kind == "global":
                blocks.append(GlobalAttention3D(dim, global_size, dim_head, mlp_ratio, drop_path))
            else:
                raise ValueError(kind)
        self.blocks = nn.Sequential(*blocks)
        self.kinds = tuple(kinds)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, GlobalAttention3D):
                for p in m.pos:
                    nn.init.trunc_normal_(p, std=.02)

    def forward(self, x):
        if any(s % 4 for s in x.shape[2:]):
            raise ValueError("attention stages need spatial sizes divisible by 4")
        x = x.permute(0, 2, 3, 4, 1)                      # channels-last for the attention blocks
        x = self.blocks(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class UNetMaxViT3D(UNet3D):
    def __init__(self, base=42, out_channels=1, attn32=("block", "grid"), attn16=("block", "global"),
                 replace=False, dim_head=24, mlp_ratio=2, drop_path32=0.1, drop_path16=0.2,
                 checkpoint_highres=False):
        super().__init__(base=base, out_channels=out_channels)
        self.checkpoint_highres = bool(checkpoint_highres)
        self.replace = bool(replace)
        if self.replace:                                  # keep conv-IN-ReLU, drop the second 3^3 conv
            self.enc3 = nn.Sequential(*list(self.enc3)[:3])
            self.mid = nn.Sequential(*list(self.mid)[:3])
        # Same seed -> the U-Net part starts from exactly UNet-L's weights.
        with torch.random.fork_rng(devices=[]):
            self.attn32 = AttnStage(base * 4, attn32, dim_head, mlp_ratio, drop_path32) if attn32 else nn.Identity()
            self.attn16 = AttnStage(base * 8, attn16, dim_head, mlp_ratio, drop_path16) if attn16 else nn.Identity()

    def _detail(self, block, x):
        if self.training and self.checkpoint_highres:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        a = self._detail(self.enc1, x)
        b = self.enc2(self.pool(a))
        c = self.attn32(self.enc3(self.pool(b)))
        x = self.attn16(self.mid(self.pool(c)))
        x = self.dec3(torch.cat((self.up3(x), c), 1))
        x = self.dec2(torch.cat((self.up2(x), b), 1))
        return self.out(self._detail(self.dec1, torch.cat((self.up1(x), a), 1)))
