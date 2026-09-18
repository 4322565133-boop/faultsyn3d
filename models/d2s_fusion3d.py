"""Residual deep-to-shallow fusion. No geometry is needed at inference."""
import torch
from torch import nn
from torch.nn import functional as F


class D2SFusion3D(nn.Module):
    def __init__(self, shallow_ch, decoder_ch, deep_channels, width=32):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv3d(c, width, 1) for c in deep_channels)
        self.scale_score = nn.Conv3d(width * len(deep_channels), len(deep_channels), 1)
        self.shallow = nn.Conv3d(shallow_ch, width, 1)
        self.decoder = nn.Conv3d(decoder_ch, width, 1)
        self.gate = nn.Sequential(nn.Conv3d(3 * width, width, 1), nn.GroupNorm(8, width),
                                  nn.SiLU(), nn.Conv3d(width, shallow_ch, 1))
        self.restore = nn.Conv3d(width, shallow_ch, 1)
        # Exact identity at initialization allows a controlled warm start.
        self.context_strength = nn.Parameter(torch.zeros(()))
        self.gate_strength = nn.Parameter(torch.zeros(()))

    def forward(self, shallow, decoder, deep):
        aligned = [F.interpolate(p(x), size=shallow.shape[2:], mode='trilinear',
                                 align_corners=False) for p, x in zip(self.projections, deep)]
        weights = self.scale_score(torch.cat(aligned, 1)).softmax(1)
        context = sum(x * weights[:, j:j+1] for j, x in enumerate(aligned))
        gate = self.gate(torch.cat([self.shallow(shallow), context, self.decoder(decoder)], 1)).sigmoid()
        return ((1 + .5 * self.gate_strength.tanh() * (2 * gate - 1)) * shallow
                + self.context_strength.tanh() * self.restore(context))
