"""FP32 segmentation, normal-profile JS, and sampled 2D soft-clDice."""
import torch
from torch.nn import functional as F


def segmentation_loss(logits, target):
    z, y = logits.float(), target.float()
    p = z.sigmoid()
    dims = (1, 2, 3, 4)
    dice = (1 - (2 * (p*y).sum(dims) + 1e-6) / (p.sum(dims) + y.sum(dims) + 1e-6)).mean()
    pt = torch.where(y > .5, p, 1-p)
    alpha = torch.where(y > .5, .75, .25)
    focal = (alpha * (1-pt).square() * F.binary_cross_entropy_with_logits(z, y, reduction='none')).mean()
    return .6*dice + .4*focal


def profile_loss(prob, target, points_zyx, normals_zyx, valid):
    offsets = torch.arange(-3, 4, device=prob.device, dtype=torch.float32)
    xyz = (points_zyx[..., None, :] + normals_zyx[..., None, :] * offsets[None, None, :, None]).flip(-1)
    size = torch.tensor(prob.shape[-3:][::-1], device=prob.device, dtype=torch.float32)
    grid = (2*(xyz+.5)/size - 1).unsqueeze(3)
    p = F.grid_sample(prob.float(), grid, align_corners=False)[:, 0, :, :, 0]
    y = F.grid_sample(target.float(), grid, align_corners=False)[:, 0, :, :, 0]
    keep = valid.bool() & (y.sum(-1) > 1e-4) & ((grid.abs() < 1).all(-1).all(-1).all(-1))
    a, b = p+1e-6, y+1e-6
    a, b = a/a.sum(-1, keepdim=True), b/b.sum(-1, keepdim=True)
    m = (a+b)*.5
    js = .5*(a*(a.log()-m.log()) + b*(b.log()-m.log())).sum(-1)
    return (js * keep).sum()/keep.sum().clamp_min(1)


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x, iterations=10):
    def opening(t):
        return F.max_pool2d(soft_erode(t), 3, 1, 1)
    skeleton = F.relu(x - opening(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - opening(x))
        skeleton = skeleton + F.relu(delta - skeleton*delta)
    return skeleton


def trace_loss(prob, target, windows):
    """windows: CPU int tensor B,K,4 = axis, plane, row, column; -1 disables."""
    pp, yy = [], []
    for b, sample_windows in enumerate(windows.tolist()):
        for axis, plane, row, col in sample_windows:
            if axis < 0:
                continue
            sl = [slice(None)]*3
            sl[axis] = plane
            pp.append(prob[b, 0][tuple(sl)][row:row+64, col:col+64])
            yy.append(target[b, 0][tuple(sl)][row:row+64, col:col+64])
    if not pp:
        return prob.sum()*0
    p, y = torch.stack(pp)[:, None].float(), torch.stack(yy)[:, None].float()
    sp, sy = soft_skeleton(p), soft_skeleton(y)
    # 10 morphology steps can propagate at most ~22 pixels; avoid crop-border effects.
    p, y, sp, sy = [t[..., 22:-22, 22:-22] for t in (p, y, sp, sy)]
    dims = (1, 2, 3)
    keep = sy.sum(dims) > 0
    precision = ((sp*y).sum(dims)+1e-6)/(sp.sum(dims)+1e-6)
    sensitivity = ((sy*p).sum(dims)+1e-6)/(sy.sum(dims)+1e-6)
    loss = 1-2*precision*sensitivity/(precision+sensitivity+1e-6)
    return (loss*keep).sum()/keep.sum().clamp_min(1)
