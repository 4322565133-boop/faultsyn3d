"""Live validation comparison: grid U-Net, paired control and historical curves."""
import csv
import html
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    if not path.exists(): return []
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r.get('val_iou')]


def render(root):
    root=Path(root)
    g=rows(root/'grid/log.csv'); b=rows(root/'baseline/log.csv')
    h=rows(root/'reference_unet_l.csv');m=rows(root/'reference_maxvit.csv')
    datasets=[(m,'MaxViT historical 40.16M','#8192a0'),(h,'UNet-L historical 9.64M (56 epochs)','#e7a232'),
              (b,'UNet-L matched control 9.64M','#2674b5'),(g,'UNet-L + grid 10.78M','#d64638')]
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    def plot(ax,r,key,color,label,ls='-'):
        valid=[v for v in r if v.get(key)]
        if valid:
            ax.plot([int(v['epoch']) for v in valid],[float(v[key]) for v in valid],
                    color=color,label=label,ls=ls,marker='.' if len(valid)<3 else None)
    for r,label,color in datasets:
        plot(axes[0,0],r,'val_iou',color,label)
        plot(axes[0,1],r,'val_loss',color,label)
    axes[0,0].set_title('Validation IoU');axes[0,0].set_ylim(0,1)
    axes[0,1].set_title('Validation Dice+Focal loss')
    for r,label,color in [(b,'Matched UNet-L','#2674b5'),(h,'Historical UNet-L','#e7a232'),(m,'Historical MaxViT','#8192a0')]:
        ref={v['epoch']:v for v in r};p=[v for v in g if v['epoch'] in ref]
        if p: axes[0,2].plot([int(v['epoch']) for v in p],[100*(float(v['val_iou'])-float(ref[v['epoch']]['val_iou'])) for v in p],
                            label='Grid minus '+label,color=color,marker='.' if len(p)<3 else None)
    axes[0,2].axhline(0,color='gray',lw=1);axes[0,2].set_title('Same-epoch IoU difference (percentage points)')
    target=['listric_assemblage','horsetail','negative_flower']
    for ax,cat in zip(axes[1],target):
        for r,label,color in datasets[2:]:
            plot(ax,r,'val_precision_'+cat,color,label+' P','--')
            plot(ax,r,'val_recall_'+cat,color,label+' R','-')
        ax.set_title(cat+': precision dashed / recall solid');ax.set_ylim(0,1)
    for ax in axes.flat:
        ax.grid(alpha=.25);ax.set_xlim(0,200);ax.set_xlabel('Epoch')
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
    status=json.loads((root/'pipeline_status.json').read_text()) if (root/'pipeline_status.json').exists() else {'state':'starting'}
    latest=g[-1] if g else None
    title='Grid attention on unchanged U-Net-L | updated '+time.strftime('%H:%M:%S')
    if latest:title+=f" | epoch {latest['epoch']}/200 | grid val IoU {float(latest['val_iou']):.4f}"
    fig.suptitle(title+'\n'+status['state']+' | historical UNet-L line ends at 56; no extrapolation',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));tmp=root/'live_curves.tmp.png';fig.savefig(tmp,dpi=110);plt.close(fig);tmp.replace(root/'live_curves.png')
    metrics={'updated':time.time(),'pipeline':status,'grid_latest':latest,'baseline_latest':b[-1] if b else None}
    for variant in ['grid','baseline']:
        p=root/variant/'status.json'
        if p.exists():metrics[variant+'_status']=json.loads(p.read_text())
    tmp=root/'live_metrics.tmp.json';tmp.write_text(json.dumps(metrics,indent=2));tmp.replace(root/'live_metrics.json')
    table=''
    if latest:
        for cat in ['en_echelon','horsetail','negative_flower','positive_flower','listric_assemblage']:
            table+=f'<tr><td>{cat}</td>'+''.join(f'<td>{float(latest[f"val_{k}_{cat}"]):.4f}</td>' for k in ['iou','precision','recall'])+'</tr>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>U-Net-L + Grid 实时对照</title>
<style>body{font:16px system-ui;max-width:1700px;margin:24px auto;padding:0 24px;background:#f6f8fa;color:#202630}p{line-height:1.7}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px 20px}</style>
<h1>原版 U-Net-L ＋两处 Grid Attention</h1><p>保留卷积、转置卷积、跳连和 loss；随机初始化。先训练 grid 版，再训练同协议原版，各 200 轮、4 卡、全局 batch 8。两模型共有权重的初值相同。历史 U-Net-L 仅有 56 轮，不能当作最终上限。</p>'''
    page+=f'<p>{html.escape(json.dumps(metrics,ensure_ascii=False))}</p><img src="live_curves.png?v={int(time.time())}">'
    page+='<h2>Grid 模型最新完成轮次：验证集类别指标</h2><table><tr><th>类别</th><th>IoU</th><th>Precision</th><th>Recall</th></tr>'+table+'</table>'
    page+='<p>重点：铲式组合的 P/R，以及马尾状、负花状的召回。P/R 仍应联合判断，不能用增加误报换取表面上的召回提升。页面每 30 秒刷新；训练期间不反复评估 test。</p></html>'
    tmp=root/'index.tmp.html';tmp.write_text(page);tmp.replace(root/'index.html')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);render(p.parse_args().root)
