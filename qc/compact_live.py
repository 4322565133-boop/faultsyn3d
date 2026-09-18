"""Atomic, offline live report against the frozen historical MaxViT curve."""
import csv
import html
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    if not Path(path).exists(): return []
    with Path(path).open() as f:
        return [r for r in csv.DictReader(f) if r.get('val_iou') and r.get('epoch')]


def render(out):
    out=Path(out)
    old=read_rows(out/'reference_maxvit_log.csv'); new=read_rows(out/'log.csv')
    status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'state':'starting'}
    by_epoch={int(r['epoch']):r for r in old}
    latest=new[-1] if new else None
    same=by_epoch.get(int(latest['epoch'])) if latest else None
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    colors=['#2879b9','#d94a36']; labels=['Baseline MaxViT (40.16M)','Compact MaxViT M0 (9.49M)']
    def plot(ax,rows,key,color,label,ls='-'):
        valid=[r for r in rows if r.get(key)]
        ax.plot([int(r['epoch']) for r in valid],[float(r[key]) for r in valid],color=color,label=label,ls=ls)
    for rows,color,label in zip([old,new],colors,labels):
        plot(axes[0,0],rows,'val_iou',color,label)
        plot(axes[0,1],rows,'val_loss',color,label+' val')
        plot(axes[0,1],rows,'train_loss',color,label+' train','--')
        plot(axes[1,0],rows,'val_precision',color,label+' precision','--')
        plot(axes[1,0],rows,'val_recall',color,label+' recall')
    axes[0,0].set_title('Validation IoU (threshold 0.5); TEST scores are not plotted')
    axes[0,0].set_ylim(0,1)
    axes[0,1].set_title('Dice + Focal loss (historical AMP vs explicit FP32)')
    axes[1,0].set_title('Validation precision / recall'); axes[1,0].set_ylim(0,1)
    paired=[(r,by_epoch[int(r['epoch'])]) for r in new if int(r['epoch']) in by_epoch]
    if paired:
        axes[1,1].plot([int(r['epoch']) for r,_ in paired],
                       [100*(float(r['val_iou'])-float(o['val_iou'])) for r,o in paired],
                       color=colors[1],label='M0 minus baseline at same epoch')
    axes[1,1].axhline(0,color='gray',lw=1);axes[1,1].set_title('Matched-epoch validation IoU difference (percentage points)')
    for ax in axes.flat:
        ax.grid(alpha=.25);ax.set_xlabel('Epoch');ax.set_xlim(0,200)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    subtitle='Waiting for first validation'
    if latest:
        subtitle=f"Epoch {latest['epoch']}/200 | M0 IoU {float(latest['val_iou']):.4f}"
        if same:subtitle+=f" | baseline same epoch {float(same['val_iou']):.4f} | delta {100*(float(latest['val_iou'])-float(same['val_iou'])):+.3f} pp"
    fig.suptitle(subtitle+'\nUpdated '+time.strftime('%Y-%m-%d %H:%M:%S')+' | '+status['state'],fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    tmp=out/'live_curves.tmp.png';fig.savefig(tmp,dpi=110);plt.close(fig);tmp.replace(out/'live_curves.png')
    metrics={'updated':time.time(),'status':status,'latest':latest,'baseline_same_epoch':same,
             'best_new':max(new,key=lambda r:float(r['val_iou'])) if new else None,
             'best_baseline':max(old,key=lambda r:float(r['val_iou'])) if old else None}
    tmp=out/'live_metrics.tmp.json';tmp.write_text(json.dumps(metrics,indent=2));tmp.replace(out/'live_metrics.json')
    catrows=''
    if same:
        for k in latest:
            if k.startswith('val_iou_'):
                catrows+=f'<tr><td>{html.escape(k[8:])}</td><td>{float(same[k]):.4f}</td><td>{float(latest[k]):.4f}</td><td>{100*(float(latest[k])-float(same[k])):+.3f}</td></tr>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>Compact MaxViT 实时训练对比</title><style>body{font-family:system-ui;max-width:1450px;margin:24px auto;padding:0 20px;background:#f6f7f9;color:#202530}img{width:100%;background:white}table{border-collapse:collapse;background:white}td,th{padding:8px 16px;border:1px solid #ddd}p{line-height:1.7}</style>
<h1>新数据集：Compact MaxViT 与 MaxViT-UNet 基线</h1>'''
    page+=f'<p><strong>{html.escape(subtitle)}</strong><br>状态：{html.escape(json.dumps(status,ensure_ascii=False))}</p>'
    page+='<p>948.9 万参数；随机初始化；4 GPU、全局 batch 8；200 epoch。历史基线是参考轨迹，单卡累积、初始化与数值执行存在差异。当前图比较验证集；历史测试 IoU 0.83259 不作为验证曲线阈值。页面每 30 秒刷新。</p>'
    page+=f'<img src="live_curves.png?v={int(time.time())}"><h2>最新完成轮次的类别对比</h2><table><tr><th>类别</th><th>基线同轮 IoU</th><th>M0 IoU</th><th>差值／百分点</th></tr>{catrows}</table></html>'
    tmp=out/'index.tmp.html';tmp.write_text(page);tmp.replace(out/'index.html')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);render(p.parse_args().out)
