"""Validation-only inference ablation; not a substitute for training ablations."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.maxvit3d_enhanced import EnhancedMaxViT3D
from train.dataset import FaultVolumes


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    run = ROOT/'runs/model_upgrade_ddp_20260916'
    ck = torch.load(run/'combined/best.pt', map_location='cpu', weights_only=False)
    model = EnhancedMaxViT3D(**ck['config']['model_kwargs']).to(device).eval()
    model.load_state_dict(ck['model'])
    ratios = {name: [] for name in ['fusion32', 'fusion64']}
    hooks = []
    for name in ratios:
        def hook(module, args, output, name=name):
            x = args[0].float()
            ratios[name].append(((output.float()-x).norm()/x.norm().clamp_min(1e-6)).item())
        hooks.append(getattr(model, name).register_forward_hook(hook))
    counts = np.zeros((2, 3), np.int64)
    prob_sum = 0.; flipped = 0; voxels = 0
    per_volume = []
    dataset = FaultVolumes(ROOT/'data/dataset_reproduction_v2', 'val')
    loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            x, truth = x.to(device), y.to(device) > .5
            model.use_d2s = True
            with torch.autocast('cuda', dtype=torch.float16):
                on = model(x).float().sigmoid()
            model.use_d2s = False
            with torch.autocast('cuda', dtype=torch.float16):
                off = model(x).float().sigmoid()
            row = {'name': dataset.items[i][0]}
            for k, (name, prob) in enumerate([('on', on), ('off', off)]):
                pred = prob > .5
                tp = (pred & truth).sum().item()
                fp = (pred & ~truth).sum().item()
                fn = (~pred & truth).sum().item()
                counts[k] += [tp, fp, fn]
                row[name+'_iou'] = tp/max(tp+fp+fn, 1)
            prob_sum += (on-off).abs().sum().item()
            flipped += ((on>.5) != (off>.5)).sum().item()
            voxels += truth.numel()
            per_volume.append(row)
            if (i+1) % 25 == 0:
                print(f'validation intervention {i+1}/{len(dataset)}', flush=True)
    for hook in hooks:
        hook.remove()
    metrics = {}
    for name, (tp, fp, fn) in zip(['on', 'off'], counts):
        metrics[name] = {'iou': float(tp/(tp+fp+fn)), 'precision': float(tp/(tp+fp)),
                         'recall': float(tp/(tp+fn)), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    results = {'split': 'val', 'n_volumes': len(dataset), 'checkpoint_epoch': ck['epoch'],
               'kind': 'inference intervention in the same trained model; not retraining ablation',
               'metrics': metrics, 'module_on_minus_off_iou': metrics['on']['iou']-metrics['off']['iou'],
               'mean_absolute_probability_change': prob_sum/voxels,
               'threshold_prediction_flip_fraction': flipped/voxels,
               'fusion_relative_l2_change': {k: {'mean': float(np.mean(v)), 'max': float(np.max(v))}
                                            for k, v in ratios.items()}, 'per_volume': per_volume}
    (run/'validation_module_intervention.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k:v for k,v in results.items() if k!='per_volume'}, indent=2))


if __name__ == '__main__':
    main()
