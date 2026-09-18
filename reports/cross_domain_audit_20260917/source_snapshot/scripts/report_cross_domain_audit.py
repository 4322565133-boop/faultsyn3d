"""Report frozen checkpoint audit, paired perturbations and observational geometry strata."""
import json,csv,html
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.audit_cross_domain import metrics,CATEGORIES,STYLES
P=Path('reports/cross_domain_audit_20260917');MODELS=['unet','maxvit','hybrid'];LABELS=['UNet-L','MaxViT','UNet + attention'];COLORS=['#2474ab','#e07824','#289c69']
def read(m,key):return json.loads((P/m/(key+'.json')).read_text())
def pooled(rows):
 c={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']};return metrics(c)
def table(rows):
 return '<table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{html.escape(str(v))}</{"th" if i==0 else "td"}>' for v in r)+'</tr>' for i,r in enumerate(rows))+'</table>'
def figsave(fig,name):fig.tight_layout();fig.savefig(P/name,dpi=150);plt.close(fig)
def main():
 for m in MODELS:assert (P/m/'complete.json').exists(),m
 allsets=['v2_val','v2_test','faultseg3d20','wu_style_test'];data={m:{s:read(m,s) for s in allsets} for m in MODELS};rows=[]
 for s in allsets:
  for m,l in zip(MODELS,LABELS):
   d=data[m][s];rows.append(dict(model=l,dataset=s,n=d['n'],epoch=read(m,'provenance')['epoch'],**d['metrics']))
  base=data['unet'][s]['volumes']
  for m in MODELS[1:]:
   q=data[m][s]['volumes'];assert [(r['name'],r['input_sha256'],r['label_sha256']) for r in base]==[(r['name'],r['input_sha256'],r['label_sha256']) for r in q]
 with (P/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,metric in zip(axes,['iou','ap']):
  for j,(m,l,c) in enumerate(zip(MODELS,LABELS,COLORS)):
   vals=[data[m][s]['metrics'][metric] for s in allsets];bars=ax.bar(np.arange(4)+(j-1)*.25,vals,width=.24,label=l,color=c);ax.bar_label(bars,fmt='%.3f',fontsize=8)
  ax.set_xticks(range(4),['v2 val','v2 test','official Wu20','local Wu-style100']);ax.set_ylim(0,1.05);ax.set_title(metric.upper());ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 figsave(fig,'domain_comparison.png')
 subset={r['name'] for r in read('unet','v2_style_'+STYLES[0])['volumes']};styles={};style_rows=[]
 for m in MODELS:
  clean=[r for r in data[m]['v2_val']['volumes'] if r['name'] in subset];styles[m]={'clean':pooled(clean)}
  for s in STYLES:
   d=read(m,'v2_style_'+s);styles[m][s]=d['metrics'];cleanmap={r['name']:r for r in clean}
   drops=np.array([r['metrics']['iou']-cleanmap[r['name']]['metrics']['iou'] for r in d['volumes']]);rng=np.random.default_rng(20260917)
   boot=np.mean(rng.choice(drops,(2000,len(drops)),replace=True),axis=1)
   style_rows.append(dict(model=m,style=s,n=len(drops),pooled_iou=d['metrics']['iou'],pooled_delta=d['metrics']['iou']-styles[m]['clean']['iou'],macro_paired_delta=float(drops.mean()),paired_ci95=np.quantile(boot,[.025,.975]).tolist()))
 (P/'style_statistics.json').write_text(json.dumps(style_rows,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,(title,ss) in zip(axes,[('Added Gaussian noise',STYLES[:2]),('Depth low-pass',STYLES[2:4]),('Phase rotation',STYLES[4:])]):
  for m,l,c in zip(MODELS,LABELS,COLORS):ax.plot(range(3),[styles[m][s]['iou'] for s in ['clean']+ss],'o-',label=l,color=c)
  ax.set_xticks(range(3),['clean']+[s.split('_')[1] for s in ss]);ax.set_title(title);ax.set_ylabel('IoU (same 25 validation volumes)');ax.grid(alpha=.3)
 axes[0].legend(fontsize=8);figsave(fig,'style_response.png')
 geometry={};corr=[];grows=[]
 fig,axes=plt.subplots(2,3,figsize=(13,8));attrs=['dip','curvature','max_slip','snr_db','freq_hz','visible_proxy']
 for ax,attr in zip(axes.flat,attrs):
  ref=data['unet']['v2_val']['volumes'];vals=np.array([r['geometry'][attr] for r in ref]);edges=np.quantile(vals,[1/3,2/3]);groups=np.digitize(vals,edges);geometry[attr]={'edges':edges.tolist(),'bins':[]}
  for b in range(3):
   ids=np.flatnonzero(groups==b);catcounts={ct:sum(ref[i]['category']==ct for i in ids) for ct in CATEGORIES};res={'n':len(ids),'category_counts':catcounts}
   for m in MODELS:res[m]=pooled([data[m]['v2_val']['volumes'][i] for i in ids])
   geometry[attr]['bins'].append(res)
  for m,l,c in zip(MODELS,LABELS,COLORS):ax.plot(range(3),[r[m]['iou'] for r in geometry[attr]['bins']],'o-',label=l,color=c)
  ax.set_title(attr+f' (cuts {edges[0]:.2f}, {edges[1]:.2f})');ax.set_xticks(range(3),['low','mid','high']);ax.set_ylabel('pooled IoU');ax.grid(alpha=.2)
  for cat in CATEGORIES:
   for m in MODELS:
    rr=[r for r in data[m]['v2_val']['volumes'] if r['category']==cat];xx=[r['geometry'][attr] for r in rr];yy=[r['metrics']['iou'] for r in rr]
    rho=float(spearmanr(xx,yy).statistic) if len(set(xx))>1 else None
    corr.append(dict(model=m,category=cat,attribute=attr,n=len(rr),spearman=rho))
 axes.flat[0].legend(fontsize=8);figsave(fig,'geometry_strata.png');(P/'geometry_statistics.json').write_text(json.dumps(dict(strata=geometry,within_category_correlations=corr),indent=2))
 fig,ax=plt.subplots(figsize=(11,4))
 for j,(m,l,c) in enumerate(zip(MODELS,LABELS,COLORS)):
  vv=[]
  for cat in CATEGORIES:
   r=pooled([r for r in data[m]['v2_val']['volumes'] if r['category']==cat]);vv.append(r['iou']);grows.append(dict(model=m,category=cat,**r))
  ax.bar(np.arange(5)+(j-1)*.25,vv,.24,label=l,color=c)
 ax.set_xticks(range(5),['en echelon','horsetail','negative flower','positive flower','listric']);ax.set_ylim(0,1);ax.set_ylabel('Validation IoU');ax.legend();figsave(fig,'category_comparison.png');(P/'category_statistics.json').write_text(json.dumps(grows,indent=2))
 # Bootstrap uncertainty over paired volumes, conditional on the fixed checkpoints.
 comparisons=[]
 for s in allsets:
  for other in ['unet','maxvit']:
   a=data['hybrid'][s]['volumes'];b=data[other][s]['volumes'];delta=np.array([x['metrics']['iou']-y['metrics']['iou'] for x,y in zip(a,b)]);rng=np.random.default_rng(17);boot=rng.choice(delta,(5000,len(delta)),replace=True).mean(1)
   comparisons.append(dict(dataset=s,contrast='hybrid-'+other,macro_mean_delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(delta)))
 (P/'paired_uncertainty.json').write_text(json.dumps(comparisons,indent=2))
 # Identical fixed samples and center crossline, not selected for model wins.
 gallery=[]
 for f in sorted((P/'unet').glob('*.npz')):
  stem=f.stem;z=np.load(f);x,y=z['seismic'],z['label'];fig,axes=plt.subplots(2,4,figsize=(13,7))
  axes[0,0].imshow(x,cmap='gray',vmin=-2.5,vmax=2.5);axes[0,0].set_title('Seismic');axes[1,0].imshow(y,cmap='gray',vmin=0,vmax=1);axes[1,0].set_title('GT')
  for j,(m,l) in enumerate(zip(MODELS,LABELS),1):
   zz=np.load(P/m/f.name);pp=zz['probability'];assert np.array_equal(zz['label'],y)
   axes[0,j].imshow(pp,cmap='magma',vmin=0,vmax=1);axes[0,j].set_title(l+' probability')
   pred=pp>.5;rgb=np.zeros((*y.shape,3));rgb[pred&y]=[.1,.8,.2];rgb[pred&~y]=[1,.1,.1];rgb[~pred&y]=[.1,.5,1]
   axes[1,j].imshow(rgb);axes[1,j].set_title('TP green / FP red / FN blue')
  for ax in axes.flat:ax.axis('off')
  fig.suptitle(stem+' | y=64, z vertical, x horizontal',fontsize=10);name='sample_'+stem+'.png';figsave(fig,name);gallery.append(name)
 headers=['数据','模型','n','epoch','IoU','AP','P','R','容差P','容差R','标签占比']
 tab=[headers]+[[r['dataset'],r['model'],r['n'],r['epoch']]+[f'{r[k]:.4f}' for k in ['iou','ap','precision','recall','tol2_precision','tol2_recall','label_fraction']] for r in rows]
 provenance=table([['模型','参数量','所选epoch','checkpoint SHA256']]+[[l,read(m,'provenance')['params'],read(m,'provenance')['epoch'],read(m,'provenance')['checkpoint_sha256'][:16]] for m,l in zip(MODELS,LABELS)])
 body='<h1>已有模型跨域诊断 · 2026-09-17</h1><p>本次只推理，不训练。固定历史 best.pt（UNet 56 / MaxViT 200 / hybrid 36 epochs），不据目标集选权重或阈值。非统一训练预算下的架构最终排名。</p>'+provenance+'<h2>1. 完整数据集评估</h2>'+table(tab)+'<img src="domain_comparison.png"><h2>2. 成像扰动</h2><p>每类前5个验证体，共25个；固定标签与随机噪声。低通/相位是沿深度方向的后处理，不是重新生成地震。图中clean为同一25个体。</p><img src="style_response.png"><h2>3. 几何与成像难度</h2><p>100个源验证体，低/中/高为等频分组。倾角、曲率和类别相互关联，图示为观察性关联，不证明因果。curvature为生成参数|beta_dip|+|beta_strike|均值，不是曲面的物理曲率；visible_proxy不是人工可见率。</p><img src="category_comparison.png"><img src="geometry_strata.png"><h2>4. 固定样本误差图</h2><p>每类首个验证样本、外部前两个样本；均为y=64切片。没有按效果挑图。绿色正确检测，红色误检，蓝色漏检。</p>'+''.join(f'<details><summary>{g}</summary><img loading="lazy" src="{g}"></details>' for g in gallery)+'<h2>数据与限制</h2><p>Wu20是官方合成验证数据；Wu-style100是本地生成数据，均非真实工区。本轮未做目标微调。严格IoU不膨胀标签；容差指标采用Chebyshev距离2，Precision与Recall分别按预测侧、真值侧计数。置信区间按体重采样，只反映当前权重的样本不确定性，不含训练种子不确定性。历史测试集已被查看，不称盲测。</p><p><a href="summary.csv">分数CSV</a> · <a href="protocol.json">固定协议</a> · <a href="paired_uncertainty.json">配对不确定性</a> · <a href="geometry_statistics.json">几何统计</a> · <a href="style_statistics.json">成像统计</a> · <a href="REPORT.md">中文结论</a></p>'
 (P/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>跨域诊断</title><style>body{font-family:system-ui;max-width:1280px;margin:30px auto;line-height:1.7;padding:20px;background:#f7f9fc;color:#203044}table{border-collapse:collapse;background:white;font-size:14px}td,th{padding:8px;border:1px solid #ccd5df}img{max-width:100%;background:white;margin:12px 0}summary{cursor:pointer;padding:8px}h2{margin-top:36px}</style>'+body)
 print('Report ready',P/'index.html')
if __name__=='__main__':main()
