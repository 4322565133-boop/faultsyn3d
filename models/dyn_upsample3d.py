"""Shallow-guided 3-D dynamic upsampling (learned sampling positions).

Replaces ONE fixed trilinear upsample in the decoder with a learned one: for
every output voxel the module predicts a small 3-D offset, and the coarse
feature map is resampled at (base position + offset) instead of at the fixed
trilinear position.  The offset is predicted from the shallow (fine) encoder
feature at the target resolution together with the trilinearly-upsampled coarse
feature, so the fine detail says *where* in the coarse map to read from.

Mechanism reference: DySample (Liu et al., ICCV 2023) learns sampling offsets in
2-D; CARAFE / FADE learn content-aware kernels.  This is the 3-D, encoder-guided
variant.  Whether learned sampling positions matter for thin, inclined fault
surfaces is exactly the hypothesis under test -- it is not assumed.

Design constraints kept deliberately simple for a clean ablation:
  * offsets are shared across channels (one 3-vector per output voxel)
  * offset head: 1x1x1 to 16 ch -> SiLU -> 3x3x3 -> 3 ch, tanh, scaled by r
  * r = 0.5 COARSE-grid voxels (about 1 target-grid voxel of freedom)
  * zero offset reproduces F.interpolate(..., 'trilinear', align_corners=False)
    to numerical tolerance (checked in the smoke test)
  * no extra residual scalar in front of the whole path: the module IS the
    upsampler, it must not be able to fade itself out
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicUpsample3D(nn.Module):
    def __init__(self, coarse_ch: int, fine_ch: int, width: int = 16, max_offset: float = 0.5):
        super().__init__()
        self.max_offset = float(max_offset)
        self.proj_fine = nn.Conv3d(fine_ch, width, 1)
        self.proj_coarse = nn.Conv3d(coarse_ch, width, 1)
        self.head = nn.Sequential(nn.SiLU(), nn.Conv3d(2 * width, width, 3, padding=1),
                                  nn.SiLU(), nn.Conv3d(width, 3, 1))
        # small random init so the offsets start near zero but are not stuck at
        # an exact zero (the gradient of tanh at 0 is 1, so zero is learnable too)
        nn.init.normal_(self.head[-1].weight, std=1e-3); nn.init.zeros_(self.head[-1].bias)
        self.last_offset_stats = None

    @staticmethod
    def _base_grid(coarse_shape, fine_shape, device, dtype):
        """Normalised sampling grid that reproduces trilinear (align_corners=False)."""
        # grid_sample wants (N, D, H, W, 3) with the LAST dim ordered (x, y, z),
        # i.e. (W, H, D) index order, values in [-1, 1] with align_corners=False:
        #   normalised = (2 * (i + 0.5) / n_out) - 1  maps output voxel centres to
        #   input coordinates identical to F.interpolate's trilinear sampling.
        axes = []
        for n_out in fine_shape:                       # (D, H, W)
            axes.append((2 * (torch.arange(n_out, device=device, dtype=dtype) + 0.5) / n_out) - 1)
        z, y, x = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        return torch.stack([x, y, z], dim=-1)          # (D, H, W, 3) as (x, y, z)

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        N, C, Dc, Hc, Wc = coarse.shape
        Df, Hf, Wf = fine.shape[2:]
        up = F.interpolate(coarse, size=(Df, Hf, Wf), mode="trilinear", align_corners=False)
        feat = torch.cat([self.proj_fine(fine), self.proj_coarse(up)], 1)
        delta = torch.tanh(self.head(feat)) * self.max_offset          # (N, 3, Df, Hf, Wf) in COARSE voxels
        # convert coarse-voxel offsets to normalised units: 2 / n_coarse per voxel, per axis
        scale = torch.tensor([2.0 / Wc, 2.0 / Hc, 2.0 / Dc], device=delta.device, dtype=delta.dtype)
        delta = delta.permute(0, 2, 3, 4, 1)                              # (N, Df, Hf, Wf, 3) as (x, y, z)?
        # head outputs channels in (z, y, x) array order; reorder to grid_sample's (x, y, z)
        delta = delta[..., [2, 1, 0]] * scale
        grid = self._base_grid((Dc, Hc, Wc), (Df, Hf, Wf), coarse.device, coarse.dtype)[None] + delta
        out = F.grid_sample(coarse, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if not self.training or torch.is_grad_enabled():
            with torch.no_grad():
                d = delta.detach().float()
                self.last_offset_stats = {
                    "mean_abs_coarse_vox": float((d / scale).abs().mean()),
                    "saturated_frac": float(((d / scale).abs() > 0.95 * self.max_offset).float().mean()),
                    "out_of_range_frac": float(((grid.detach().abs() > 1).any(-1)).float().mean()),
                }
        return out
