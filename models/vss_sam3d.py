"""Scaled-down VSS-SAM++ (Lv et al., TMI 2026) for 3-D seismic fault segmentation.

Two independent feature pathways fused late:

  SAM branch    frozen SAM-2 Hiera-tiny image encoder (timm `sam2_hiera_tiny.fb_r896`, 26.8M, ImageNet/SA-V
                weights), applied to every vertical section (2-D, slices stacked into the batch), multi-scale
                features at strides 4/8/16/32; stacked back to 3-D and averaged over r slices (isotropic).
  Mamba branch  light 3-D conv stem (128^3 -> 16^3) + visual state-space (Mamba-2 SSD) blocks at 16^3 and 8^3
                scanning the whole volume in 4 orders (two rasters x forward/backward): cross-slice long-range 3-D context.
  Fusion        gated hybrid attention at 8^3 and 16^3: per-token channel gates on each branch ->
                cross-attention (SAM tokens query Mamba tokens) -> self-attention -> MLP  (paper Eqs. 9-11).
  Decoder       3-D U-Net decoder (conv-IN-ReLU DoubleConv) with SAM stride-4 skip, Mamba-stem skips and a
                full-resolution conv skip -> logits.

Only the Mamba branch, fusion and decoder are trained; the SAM encoder is frozen (no grad, eval mode).
The selective scan is the exact chunked SSD algorithm of Mamba-2 in pure PyTorch (no CUDA extension).
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .unet3d import DoubleConv


# ----------------------------------------------------------------------------- Mamba-2 SSD (pure PyTorch)
def _segsum(x):
    """x (..., T) -> (..., T, T) with out[i, j] = sum_{j < k <= i} x[k] (-inf above the diagonal)."""
    T = x.shape[-1]
    x = x[..., None].expand(*x.shape, T)                             # (..., T, T): x[..., k, j] = x[k]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    s = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return s.masked_fill(~mask, float("-inf"))


def ssd(X, A, B, C, block=64):
    """Chunked state-space dual scan.  X (b,l,h,p) inputs (already scaled by dt), A (b,l,h) log-decays (<=0),
    B, C (b,l,h,n).  Returns Y (b,l,h,p).  Exact; O(l * block) memory."""
    b, l, h, p = X.shape; n = B.shape[-1]
    assert l % block == 0
    c = l // block
    X = X.reshape(b, c, block, h, p); B = B.reshape(b, c, block, h, n); C = C.reshape(b, c, block, h, n)
    A = A.reshape(b, c, block, h).permute(0, 3, 1, 2)                # (b,h,c,l)
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(_segsum(A))                                        # (b,h,c,l,l) intra-chunk decays
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)          # (b,h,c,l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)      # (b,c+1,h,p,n)
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[..., -1], (1, 0))))       # (b,h,c+1,c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, new_states, torch.exp(A_cumsum))
    return (Y_diag + Y_off).reshape(b, l, h, p)


class VSSBlock3D(nn.Module):
    """Mamba-2 style block on a (B,C,D,H,W) map: LN -> in_proj -> depthwise 3x3x3 conv -> 4-direction SSD scan
    (sum) -> gate -> out_proj, + MLP.  Scans: z-major raster and x-major raster, each forward and backward."""

    def __init__(self, dim, d_state=16, head_dim=16, expand=2, mlp_ratio=2, drop_path=0.):
        super().__init__()
        self.dim = dim; self.inner = dim * expand; self.heads = self.inner // head_dim; self.hd = head_dim; self.n = d_state
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.inner + 2 * d_state + self.heads)         # z, x, B, C, dt
        self.conv = nn.Conv3d(self.inner + 2 * d_state, self.inner + 2 * d_state, 3, padding=1, groups=self.inner + 2 * d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.heads + 1, dtype=torch.float32)))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.rand(self.heads) * (0.1 - 0.001) + 0.001)))
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm_y = nn.LayerNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path <= 0:
            return x
        keep = torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def _scan(self, x, Bm, Cm, dt, order):
        """x (b,L,h,p), Bm/Cm (b,L,h,n), dt (b,L,h) already in raster `order` (a permutation of token index)."""
        A = -torch.exp(self.A_log.float())                                                # (h,)
        out = 0
        for flip in (False, True):
            xo, Bo, Co, do = [t[:, order] for t in (x, Bm, Cm, dt)]
            if flip:
                xo, Bo, Co, do = [t.flip(1) for t in (xo, Bo, Co, do)]
            y = ssd((xo * do[..., None]).float(), (do * A).float(), Bo.float(), Co.float())
            if flip:
                y = y.flip(1)
            inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
            out = out + y[:, inv]
        return out

    def forward(self, f):
        b, c, d, h, w = f.shape
        L = d * h * w
        u = f.flatten(2).transpose(1, 2)                                                    # (b,L,C) z-major raster
        t = self.in_proj(self.norm1(u))
        z, xBC, dt = torch.split(t, [self.inner, self.inner + 2 * self.n, self.heads], dim=-1)
        xBC = self.conv(xBC.transpose(1, 2).reshape(b, -1, d, h, w)).flatten(2).transpose(1, 2)
        xBC = F.silu(xBC)
        x, Bm, Cm = torch.split(xBC, [self.inner, self.n, self.n], dim=-1)
        x = x.reshape(b, L, self.heads, self.hd)
        Bm = Bm[:, :, None, :].expand(b, L, self.heads, self.n); Cm = Cm[:, :, None, :].expand(b, L, self.heads, self.n)
        dt = F.softplus(dt + self.dt_bias)                                                  # (b,L,h)
        idx = torch.arange(L, device=f.device).reshape(d, h, w)
        orders = (idx.flatten(), idx.permute(2, 1, 0).flatten())                            # z-major, x-major
        y = sum(self._scan(x, Bm, Cm, dt, o) for o in orders) / (2 * len(orders))
        y = y.to(x.dtype) + x * self.D.to(x.dtype)[None, None, :, None]
        y = self.norm_y(y.reshape(b, L, self.inner)) * F.silu(z)
        u = u + self._dp(self.out_proj(y))
        u = u + self._dp(self.mlp(self.norm2(u)))
        return u.transpose(1, 2).reshape(b, c, d, h, w)


# ----------------------------------------------------------------------------- gated hybrid attention fusion
class GatedFusion(nn.Module):
    """Paper Eqs. 9-11 at one scale: channel gates per token on each branch, cross-attention (SAM queries Mamba),
    self-attention over the fused tokens, MLP.  Returns (B, dim, D, H, W)."""

    def __init__(self, c_sam, c_mamba, dim, heads=4, mlp_ratio=2, self_attn=True):
        super().__init__()
        self.ps = nn.Conv3d(c_sam, dim, 1); self.pm = nn.Conv3d(c_mamba, dim, 1)
        self.gate_s = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim), nn.Sigmoid())
        self.gate_m = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim), nn.Sigmoid())
        self.nq = nn.LayerNorm(dim); self.nkv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = self_attn
        if self_attn:
            self.ns = nn.LayerNorm(dim); self.selfa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, S, M):
        b, _, d, h, w = S.shape
        s = self.ps(S).flatten(2).transpose(1, 2); m = self.pm(M).flatten(2).transpose(1, 2)      # (b,L,dim)
        s = s * self.gate_s(s); m = m * self.gate_m(m)
        q = self.nq(s); kv = self.nkv(m)
        f = s + self.cross(q, kv, kv, need_weights=False)[0]
        if self.self_attn:
            n = self.ns(f); f = f + self.selfa(n, n, n, need_weights=False)[0]
        f = f + self.mlp(self.nm(f))
        return f.transpose(1, 2).reshape(b, -1, d, h, w)


# ----------------------------------------------------------------------------- the model
class VSSSAM3D(nn.Module):
    def __init__(self, sam="sam2_hiera_tiny", pretrained=True, freeze_sam=True, in_slices=1, mamba=True,
                 mamba_dims=(16, 32, 64, 128), mamba_depth=(2, 2), d_state=16, fuse_dim=(128, 256), fuse_self=True,
                 dec=(128, 64, 32, 16), full_ch=16, out_channels=1, drop_path=0.1):
        super().__init__()
        assert in_slices in (1, 3)
        self.in_slices = in_slices; self.freeze_sam = bool(freeze_sam); self.use_mamba = bool(mamba)
        self.sam = timm.create_model(sam, pretrained=bool(pretrained), features_only=True, in_chans=in_slices, out_indices=(0, 1, 2))
        self.sam_ch = list(self.sam.feature_info.channels())[:3]; self.sam_red = list(self.sam.feature_info.reduction())[:3]   # 96/192/384 @ 4/8/16
        if self.freeze_sam:
            for q in self.sam.parameters():
                q.requires_grad_(False)
        m0, m1, m2, m3 = mamba_dims
        # Mamba branch: conv stem 128 -> 64 -> 32 -> 16, VSS at 16^3, downsample, VSS at 8^3
        self.stem1 = nn.Sequential(nn.Conv3d(1, m0, 3, 2, 1, bias=False), nn.InstanceNorm3d(m0), nn.ReLU(inplace=True))     # 64^3
        self.stem2 = nn.Sequential(nn.Conv3d(m0, m1, 3, 2, 1, bias=False), nn.InstanceNorm3d(m1), nn.ReLU(inplace=True))    # 32^3
        self.stem3 = nn.Sequential(nn.Conv3d(m1, m2, 3, 2, 1, bias=False), nn.InstanceNorm3d(m2), nn.ReLU(inplace=True))    # 16^3
        if self.use_mamba:
            self.vss16 = nn.Sequential(*[VSSBlock3D(m2, d_state, drop_path=drop_path) for _ in range(mamba_depth[0])])
            self.down = nn.Sequential(nn.Conv3d(m2, m3, 3, 2, 1, bias=False), nn.InstanceNorm3d(m3), nn.ReLU(inplace=True))  # 8^3
            self.vss8 = nn.Sequential(*[VSSBlock3D(m3, d_state, drop_path=drop_path) for _ in range(mamba_depth[1])])
            self.fuse16 = GatedFusion(self.sam_ch[1], m2, fuse_dim[0], heads=4, self_attn=fuse_self)
            self.fuse8 = GatedFusion(self.sam_ch[2], m3, fuse_dim[1], heads=8, self_attn=fuse_self)
            c8, c16 = fuse_dim[1], fuse_dim[0]
        else:                                             # ablation: SAM features only (1x1 projections)
            self.proj16 = nn.Conv3d(self.sam_ch[1], fuse_dim[0], 1); self.proj8 = nn.Conv3d(self.sam_ch[2], fuse_dim[1], 1)
            self.pool8 = nn.Conv3d(self.sam_ch[2], self.sam_ch[2], 1)
            c8, c16 = fuse_dim[1], fuse_dim[0]
        # decoder 8 -> 16 -> 32 -> 64 -> 128
        self.enc_full = DoubleConv(1, full_ch)
        self.dec16 = DoubleConv(c8 + c16, dec[0])
        self.dec32 = DoubleConv(dec[0] + self.sam_ch[0] + m1, dec[1])
        self.dec64 = DoubleConv(dec[1] + m0, dec[2])
        self.dec128 = DoubleConv(dec[2] + full_ch, dec[3])
        self.out = nn.Conv3d(dec[3], out_channels, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_sam:
            self.sam.eval()                              # frozen encoder stays in eval mode
        return self

    def _slices(self, x):
        B, _, D, H, W = x.shape
        if self.in_slices == 3:
            xp = F.pad(x, (1, 1), mode="replicate"); x = torch.cat((xp[..., :-2], xp[..., 1:-1], xp[..., 2:]), 1)
        return x.permute(0, 4, 1, 2, 3).reshape(B * W, x.shape[1], D, H)

    def _sam_features(self, x):
        B, _, D, H, W = x.shape
        ctx = torch.no_grad() if self.freeze_sam else torch.enable_grad()
        with ctx:
            feats = self.sam(self._slices(x))
        out = []
        for f, r in zip(feats, self.sam_red):
            f = f.reshape(B, W, f.shape[1], f.shape[2], f.shape[3]).permute(0, 2, 3, 4, 1)   # (B,C,D/r,H/r,W)
            out.append(F.avg_pool3d(f, (1, 1, r)))                                            # isotropic (B,C,D/r,H/r,W/r)
        return out                                                                            # strides 4, 8, 16

    def forward(self, x):
        s4, s8, s16 = self._sam_features(x)
        a = self.stem1(x); b = self.stem2(a); c = self.stem3(b)                             # 64^3, 32^3, 16^3
        if self.use_mamba:
            m16 = self.vss16(c); m8 = self.vss8(self.down(m16))
            f16 = self.fuse16(s8, m16); f8 = self.fuse8(s16, m8)
        else:
            f16 = self.proj16(s8); f8 = self.proj8(s16)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="trilinear", align_corners=False)
        y = self.dec16(torch.cat((up(f8), f16), 1))
        y = self.dec32(torch.cat((up(y), s4, b), 1))
        y = self.dec64(torch.cat((up(y), a), 1))
        y = self.dec128(torch.cat((up(y), self.enc_full(x)), 1))
        return self.out(y)
