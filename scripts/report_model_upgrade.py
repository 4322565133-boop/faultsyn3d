"""Summarize completed matched runs, plot histories and fixed three-column examples."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.maxvit3d_enhanced import EnhancedMaxViT3D
from train.dataset import FaultVolumes, CATEGORIES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs/model_upgrade_ddp_20260916')
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    root = Path(a.root)
    runs = {name: json.loads((root/name/'test.json').read_text()) for name in ['baseline', 'combined']}
    reference = json.loads((ROOT/'runs/maxvit3d_v2_oldrecipe_all3/test.json').read_text())['iou']
    delta_old = runs['combined']['iou']-reference
    delta_control = runs['combined']['iou']-runs['baseline']['iou']
    report = {'historical_iou': reference, 'runs': runs, 'delta_vs_historical': delta_old,
              'delta_vs_matched_control': delta_control, 'kind': 'single-seed warm-start four-GPU experiment'}
    (root/'comparison.json').write_text(json.dumps(report, indent=2))
    conclusion = ('本轮改进模型同时超过历史结果与同预算基线。' if delta_old > 0 and delta_control > 0 else
                  '本轮超过历史结果，但没有超过同预算基线，不能把提升归因于新方法。' if delta_old > 0 else
                  '本轮没有超过历史测试IoU，尚不能宣称模型改进有效。')
    lines = ['# 四卡 MaxViT 升级试验结果', '', conclusion, '',
             '同一历史checkpoint继续训练；同一20轮预算、数据划分和验证集选模；单训练种子。', '',
             '| 模型 | 最佳继续训练轮次 | Test IoU | Dice | Precision | Recall |',
             '|---|---:|---:|---:|---:|---:|',
             f'| 历史MaxViT | — | {reference:.6f} | — | — | — |']
    for name, r in runs.items():
        lines.append(f'| {name} | {r["epoch"]} | {r["iou"]:.6f} | {r["dice"]:.6f} | '
                     f'{r["precision"]:.6f} | {r["recall"]:.6f} |')
    lines += ['', f'相对历史提升：{100*delta_old:+.3f} 个百分点。',
              f'相对同预算基线提升：{100*delta_control:+.3f} 个百分点。', '',
              '这些结果只支持本次继续训练设定；不是从零训练、多种子或显著性结论。', '',
              '固定可视化：每类第一个测试体、中央inline/crossline切片，未按预测效果挑选。',
              '三列为输入+GT、同预算基线、改进模型；误差颜色绿=TP，红=FP，蓝=FN。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in runs:
        rows = list(csv.DictReader(open(root/name/'log.csv')))
        axes[0].plot([int(r['epoch']) for r in rows], [float(r['val_iou']) for r in rows], label=name)
        axes[1].plot([int(r['epoch']) for r in rows], [float(r['seg']) for r in rows], label=name)
    axes[0].set_title('Validation IoU (not test IoU)'); axes[1].set_title('Training segmentation loss')
    for ax in axes:
        ax.set_xlabel('Continuation epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/'training_curves.png', dpi=160); plt.close(fig)
    device = torch.device(f'cuda:{a.gpu}')
    torch.set_num_threads(2)
    data = FaultVolumes(ROOT/'data/dataset_reproduction_v2', 'test')
    selected = [next(i for i, (_, c) in enumerate(data.items) if c == cat) for cat in CATEGORIES]
    predictions = {}
    for name in runs:
        ck = torch.load(root/name/'best.pt', map_location='cpu', weights_only=False)
        model = EnhancedMaxViT3D(**ck['config']['model_kwargs']).to(device).eval()
        model.load_state_dict(ck['model'])
        with torch.no_grad():
            for i in selected:
                x, _, _ = data[i]
                with torch.autocast('cuda', dtype=torch.float16):
                    z = model(x[None].to(device))
                predictions[name, i] = (z.float().sigmoid()[0, 0] > .5).cpu().numpy()
        del model, ck
        torch.cuda.empty_cache()
    fig, axes = plt.subplots(10, 3, figsize=(12, 35))
    for j, i in enumerate(selected):
        x, y, _ = data[i]; x, y = x[0].numpy(), y[0].numpy() > .5
        for view, axis in enumerate([1, 2]):
            row = 2*j+view
            sl = [slice(None)]*3; sl[axis] = 64; sl = tuple(sl)
            image, truth = x[sl], y[sl]
            for col, name in enumerate(['GT', 'baseline', 'combined']):
                ax = axes[row, col]; ax.imshow(image, cmap='gray', vmin=-2, vmax=2)
                if col == 0:
                    ax.contour(truth, levels=[.5], colors=['gold'], linewidths=.6)
                else:
                    pred = predictions[name, i][sl]
                    overlay = np.zeros((*truth.shape, 4))
                    overlay[pred & truth] = [0, 1, 0, .7]
                    overlay[pred & ~truth] = [1, 0, 0, .8]
                    overlay[~pred & truth] = [0, .3, 1, .8]
                    ax.imshow(overlay)
                ax.set_title(f'{data.items[i][0]} / axis {axis} / {name}', fontsize=9)
                ax.axis('off')
    fig.tight_layout(); fig.savefig(root/'comparison_3col.png', dpi=130); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
