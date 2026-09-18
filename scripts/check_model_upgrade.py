"""Numerical/geometry checks and optional GPU checkpoint identity check."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.geometry_losses import segmentation_loss, profile_loss, trace_loss
from train.geometry_dataset import GeometryVolumes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=None)
    a = p.parse_args()
    torch.set_num_threads(2)
    result = {}
    z = torch.zeros(1, 1, 128, 128, 128, requires_grad=True)
    target = torch.zeros_like(z); target[..., 64] = 1
    ref = segmentation_loss(z, target)
    half = z.detach().half().requires_grad_()
    actual = segmentation_loss(half, target)
    actual.backward(); ref.backward()
    assert torch.isfinite(actual) and torch.isfinite(half.grad).all()
    assert torch.allclose(actual, ref, atol=1e-6)
    assert torch.allclose(half.grad.float(), z.grad, atol=3e-8, rtol=.15)
    result['fp16_vs_fp32_loss'] = [actual.item(), ref.item()]
    y = torch.zeros(1, 1, 32, 32, 32); y[..., 16] = 1
    q = torch.tensor([[[16., 16., 16.], [8., 8., 16.]]])
    n = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    match = y*.98+.01
    thick = y.clone(); thick[..., 14:19] = 1; thick = thick*.98+.01
    shift = torch.roll(match, 2, -1)
    lp = [profile_loss(v, y, q, n, valid).item() for v in [match, thick, shift]]
    assert lp[0] < lp[1] and lp[0] < lp[2]
    assert torch.allclose(profile_loss(match, y, q, n, valid), profile_loss(match, y, q, -n, valid))
    empty = profile_loss(match, y, q, n, torch.zeros_like(valid))
    assert empty.item() == 0
    shifted = shift.detach().requires_grad_()
    profile_loss(shifted, y, q, n, valid).backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
    result['profile_match_thick_shift'] = lp
    y2 = torch.zeros(1, 1, 64, 64, 64); y2[..., 32] = 1
    good = .98*y2+.01
    broken = good.clone(); broken[:, :, :, 26:38, 32] = .01
    win = torch.tensor([[[0, 32, 0, 0]]])
    lt = [trace_loss(v, y2, win).item() for v in [good, broken]]
    assert lt[0] < lt[1]
    broken.requires_grad_(); trace_loss(broken, y2, win).backward()
    assert torch.isfinite(broken.grad).all() and broken.grad.abs().sum() > 0
    assert trace_loss(good, y2*0, win).item() == 0
    result['trace_match_gap'] = lt
    data = GeometryVolumes(ROOT/'data/dataset_reproduction_v2', ROOT/'data/geometry_model_upgrade_v1')
    checks = []
    for epoch, i in [(1, 0), (2, 175), (3, 350), (4, 525), (5, 799)]:
        data.epoch = epoch
        x, yy, _, points, normals, mask, windows = data[i]
        qi = points[mask].long()
        occupancy = yy[0, qi[:, 0], qi[:, 1], qi[:, 2]]
        assert occupancy.min() == 1, 'Augmentation misaligns geometry and labels'
        assert torch.allclose(normals[mask].norm(dim=1), torch.ones(mask.sum()), atol=1e-5)
        grid = (2*(points[mask].flip(-1)+.5)/128 - 1)[None, :, None, None]
        center = F.grid_sample(yy[None], grid, align_corners=False).flatten()
        assert center.min() > .9999
        checks.append({'sample': i, 'epoch': epoch, 'valid_windows': int((windows[:, 0] >= 0).sum())})
    result['augmentation_checks'] = checks
    if a.gpu is not None:
        from models.maxvit3d_enhanced import EnhancedMaxViT3D
        device = torch.device(f'cuda:{a.gpu}')
        ck = torch.load(ROOT/'runs/maxvit3d_v2_oldrecipe_all3/best.pt', map_location='cpu', weights_only=False)
        base = EnhancedMaxViT3D(d2s=False, full_res_skip=True, img_size=128, drop_path_rate=.2).to(device).eval()
        enhanced = EnhancedMaxViT3D(d2s=True, full_res_skip=True, img_size=128, drop_path_rate=.2).to(device).eval()
        base.load_state_dict(ck['model']); incompatible = enhanced.load_state_dict(ck['model'], strict=False)
        assert all(k.startswith(('fusion32.', 'fusion64.')) for k in incompatible.missing_keys)
        sample = x[None].to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            before, after = base(sample), enhanced(sample)
        difference = (before.float()-after.float()).abs().max().item()
        assert difference < 1e-4, difference
        result['initial_model_max_logit_difference'] = difference
        result['parameter_counts'] = {'baseline': sum(p.numel() for p in base.parameters()),
                                      'enhanced': sum(p.numel() for p in enhanced.parameters())}
    dest = ROOT/'docs'/('MODEL_UPGRADE_CHECKS_GPU_20260916.json' if a.gpu is not None else 'MODEL_UPGRADE_CHECKS_CPU_20260916.json')
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
