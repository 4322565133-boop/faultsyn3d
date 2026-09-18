"""Static browser for every volume, plus statistics and unpainted seismic QC."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/faultsyn_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qc.render3d import best_slices


def thumbnail(job):
    root,out,row=job;root=Path(root);out=Path(out);name=row['name']
    meta=json.loads((root/'metadata'/f'{name}.json').read_text())
    nx,ny,nz=meta['grid'];s=np.fromfile(root/'seismic'/f'{name}.dat',np.float32).reshape(nz,ny,nx)
    l=np.fromfile(root/'labels'/f'{name}.dat',np.uint8).reshape(s.shape)
    ix,iy,iz=best_slices(l)
    slices=[(s[:,iy,:],l[:,iy,:]),(s[:,:,ix],l[:,:,ix]),(s[iz],l[iz])]
    tops=[];bottoms=[]
    for a,b in slices:
        rgb=np.repeat((255*np.clip((a+2.5)/5,0,1)).astype(np.uint8)[...,None],3,axis=2)
        tops.append(rgb);v=rgb.copy();v[b>0]=[237,51,33];bottoms.append(v)
    pic=np.concatenate([np.concatenate(tops,axis=1),np.concatenate(bottoms,axis=1)],axis=0)
    Image.fromarray(pic).save(out/'slices'/f'{name}.png')
    return dict(name=name,category=row['category'],split=row['split'],fault_percent=100*meta['fault_fraction'],
                attempts=meta['accepted_attempt'],planes=[ix,iy,iz],proxy=meta['label_visible_fraction'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'slices').mkdir(exist_ok=True)
    manifest=json.loads((a.root/'manifest.json').read_text());summary=json.loads((a.root/'summary.json').read_text());validation=json.loads((a.root/'validation.json').read_text())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows=list(ex.map(thumbnail,[(str(a.root),str(a.out),r) for r in manifest['volumes']]))
    (a.out/'browse_manifest.json').write_text(json.dumps(rows,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cats=list(summary['categories']);old=[1.187,1.301,1.420,1.291,.943]
    new=[summary['categories'][c]['fault_percent_mean'] for c in cats]
    labels=['En echelon','Horsetail','Negative flower','Positive flower','Listric']
    fig,axs=plt.subplots(1,2,figsize=(12,4.4));x=np.arange(5)
    axs[0].bar(x-.2,old,.4,label='Previous filtered labels',color='#93999f');axs[0].bar(x+.2,new,.4,label='Reproduction v2 full geometry',color='#267fa3')
    axs[0].set_xticks(x,labels,rotation=15);axs[0].set_ylabel('Fault voxels (%)');axs[0].legend(fontsize=8);axs[0].set_title('200 volumes per category; 128 cubed')
    axs[1].boxplot([[r['fault_percent'] for r in rows if r['category']==c] for c in cats],tick_labels=labels,showfliers=True)
    axs[1].tick_params(axis='x',rotation=15);axs[1].set_ylabel('Fault voxels (%)');axs[1].set_title('Full distribution of the 1000 new volumes')
    fig.tight_layout();fig.savefig(a.out/'statistics.png',dpi=170);plt.close(fig)
    # Chosen samples use the same indices as the main random gallery.
    picked=(a.out/'picked.txt').read_text().split() if (a.out/'picked.txt').exists() else [next(r['name'] for r in rows if r['category']==c) for c in cats]
    fig,axs=plt.subplots(len(picked),4,figsize=(12,2.6*len(picked)),squeeze=False)
    for axes,name in zip(axs,picked):
        row=next(r for r in rows if r['name']==name);ix,iy,iz=row['planes']
        s=np.fromfile(a.root/'seismic'/f'{name}.dat',np.float32).reshape(128,128,128)[:,iy,:]
        l=np.fromfile(a.root/'labels'/f'{name}.dat',np.uint8).reshape(128,128,128)[:,iy,:]
        c=np.fromfile(a.root/'confidence'/f'{name}.dat',np.uint8).reshape(128,128,128)[:,iy,:]
        axes[0].imshow(s,cmap='gray',vmin=-2.5,vmax=2.5);axes[0].set_title(f'{name}\ny={iy}: seismic only',fontsize=8)
        axes[1].imshow(l,cmap='gray',vmin=0,vmax=1);axes[1].set_title('Full geometric label',fontsize=8)
        axes[2].imshow(s,cmap='gray',vmin=-2.5,vmax=2.5);axes[2].contour(l,levels=[.5],colors=['#ed3321'],linewidths=.5);axes[2].set_title('Label boundary overlay',fontsize=8)
        axes[3].imshow(np.ma.masked_where(l==0,c/255),cmap='viridis',vmin=0,vmax=1);axes[3].set_title('Isolated-slip proxy (0 to 1)',fontsize=8)
        for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(a.out/'seismic_label_check.png',dpi=140);plt.close(fig)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Faultsyn3D 复现数据检查</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1180px;padding:0 18px;background:#f5f7f8;color:#172c35}h1{font-size:28px}button,select,input{font:inherit;padding:7px;margin:4px;border:1px solid #abbec6;border-radius:5px}img{max-width:100%;height:auto}.sample{background:white;padding:18px;margin:20px 0;border-radius:8px}.sample img{width:768px;image-rendering:auto}.cols{display:grid;grid-template-columns:repeat(3,1fr);max-width:768px;text-align:center}.muted{color:#526b75}nav{position:sticky;top:0;background:#f5f7f8;padding:8px 0;border-bottom:1px solid #bbcbd2}a{color:#116888}</style>
<h1>Faultsyn3D：1000 个体的复现结果</h1>
<p>5 类各 200 个，128³；train / val / test = 800 / 100 / 100。完整几何标签，半厚度 0.75 体素。</p>
<p><a href="overview_3col.png">五类首个样本 · 三列总览</a> · <a href="gallery_3col.png">固定随机种子 7 · 十个样本三列图</a> · <a href="seismic_label_check.png">地震与标签分开检查</a> · <a href="network_sections.png">树与正交曲面剖面</a> · <a href="statistics.png">全量统计</a> · <a href="../../../docs/reproduction_v2.md">方法与复现边界</a></p>
<p class="muted">这里检查全部样本的切片。每个体三列依次为 inline、crossline、time；上排是未叠标签的地震信号，下排是相同信号与红色标签。切片按统一的清晰断层迹线评分选择，不能代表所有位置均可见。位移代理量不是图像实测识别率。</p>
<img src="statistics.png" alt="全量统计">
<nav><label>类别 <select id="cat"><option value="">全部</option></select></label><label>划分 <select id="split"><option value="">全部</option><option>train</option><option>val</option><option>test</option></select></label><input id="search" placeholder="输入样本编号"><button id="prev">上一页</button><button id="next">下一页</button><span id="status"></span></nav><main id="cards"></main>
<script>const data=DATA;const cats=[...new Set(data.map(x=>x.category))];let page=0;
for(const c of cats){const o=document.createElement('option');o.value=c;o.textContent=c;document.querySelector('#cat').append(o)}
function render(){const cat=document.querySelector('#cat').value,split=document.querySelector('#split').value,q=document.querySelector('#search').value;const rows=data.filter(x=>(!cat||x.category===cat)&&(!split||x.split===split)&&x.name.includes(q));const pages=Math.max(1,Math.ceil(rows.length/10));page=Math.max(0,Math.min(page,pages-1));document.querySelector('#status').textContent=`${rows.length} 个体 · ${page+1}/${pages} 页`;document.querySelector('#cards').innerHTML=rows.slice(page*10,page*10+10).map(x=>`<article class="sample"><h3>${x.name}</h3><p>${x.split} · 断层体素 ${x.fault_percent.toFixed(3)}% · 接收于第 ${x.attempts} 次候选 · 切片 x/y/z=${x.planes.join('/')}</p><div class="cols"><span>Inline</span><span>Crossline</span><span>Time</span></div><img loading="lazy" src="slices/${x.name}.png" alt="${x.name} 原始信号及标签叠加"></article>`).join('')}
for(const id of ['cat','split','search'])document.getElementById(id).addEventListener('input',()=>{page=0;render()});document.querySelector('#prev').onclick=()=>{page--;render()};document.querySelector('#next').onclick=()=>{page++;render()};render();</script></html>'''
    (a.out/'index.html').write_text(html.replace('const data=DATA;', 'const data='+json.dumps(rows)+';'))
    print(a.out/'index.html',flush=True)
if __name__=='__main__':main()
