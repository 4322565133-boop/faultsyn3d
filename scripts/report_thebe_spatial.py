"""Protocol diagrams, train/val annotation audit; never opens test labels."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from train.dataset_thebe_spatial import SpatialThebe

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--figures-only',action='store_true'); args=parser.parse_args()
    out=ROOT/'reports/thebe_spatial_v3'; out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,3))
    for a,b,color,name in [(0,868,'#4c78a8','TRAIN: 868'),(868,900,'#bdbdbd',''),
                           (900,1068,'#f2b447','VAL: 168'),(1068,1100,'#bdbdbd',''),
                           (1100,1803,'#59a14f','TEST: 703 (sealed)')]:
        ax.add_patch(Rectangle((a,0),b-a,1,color=color))
        if name: ax.text((a+b)/2,.5,name,ha='center',va='center',fontsize=10)
    ax.set(xlim=(0,1803),ylim=(0,1),yticks=[],xlabel='Raw axis 0: global section index (zero based)',
        title='Thebe spatial v3 | 32-section buffers | split BEFORE patch extraction')
    ax.set_xticks([0,868,900,1068,1100,1803]); ax.tick_params(axis='x',labelrotation=45)
    fig.tight_layout(); fig.savefig(out/'split.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3))
    for row,start in enumerate((-32,32,96)):
        ax.add_patch(Rectangle((start,row),128,.7,facecolor='#d7e6f5',edgecolor='#4c78a8'))
        ax.add_patch(Rectangle((start+32,row),64,.7,facecolor='#4c78a8'))
        ax.text(start+64,row+.35,'64 scored',ha='center',va='center',color='white')
    ax.set(xlim=(-40,230),ylim=(-.2,3),yticks=[],xlabel='Voxel index along any axis',
        title='128 input = 32 context + 64 scored core + 32 context')
    fig.tight_layout(); fig.savefig(out/'patch_cores.png',dpi=160); plt.close(fig)
    train=SpatialThebe('train',samples=2400); val=SpatialThebe('val',fast=True)
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for row,(ds,ids) in enumerate([(train,[5,51,101]),(val,[24,48,72])]):
        for col,i in enumerate(ids):
            x,y,m=[t.numpy()[0] for t in ds[i]]; ax=axes[row,col]
            s=x[:,64,:]; lab=y[:,64,:]; mask=m[:,64,:]
            ax.imshow(s,cmap='gray',vmin=-2,vmax=2)
            overlay=np.zeros((*lab.shape,4)); overlay[lab>.5]=[1,.1,.1,.7]
            ax.imshow(overlay)
            if row==1: ax.add_patch(Rectangle((31.5,31.5),64,64,fill=False,edgecolor='yellow',linewidth=1))
            legend='red: released labels'+('; yellow: scored core' if row==1 else '')
            ax.set_title(f'{ds.split} deterministic sample {i}\n{legend}',fontsize=9)
            ax.set(xlabel='Horizontal index in input patch',ylabel='Sample / depth index')
    fig.tight_layout(); fig.savefig(out/'patch_examples.png',dpi=150); plt.close(fig)
    if args.figures_only: return
    audit={}
    for split in ('train','val'):
        ds=SpatialThebe(split); y=np.load(ds.record['labels'],mmap_mode='r')
        x=np.load(ds.record['seismic'],mmap_mode='r'); valid_count=positive=outside=nonfinite=0
        for z in range(0,ds.shape[0],32):
            zz=np.arange(z,min(z+32,ds.shape[0]))[:,None,None]
            m=(zz>=ds.first)&(zz<ds.last)
            yy=np.asarray(y[z:z+32,:,:ds.shape[2]])>0
            ss=np.asarray(x[z:z+32,:,:ds.shape[2]])
            valid_count+=int(m.sum()); positive+=int((yy&m).sum()); outside+=int((yy&~m).sum())
            nonfinite+=int((~np.isfinite(ss)&m).sum())
        audit[split]=dict(shape=ds.shape,valid_voxels=valid_count,positive_voxels=positive,
            positive_fraction=positive/max(valid_count,1),positive_outside_signal=outside,nonfinite_inside_signal=nonfinite)
        print(split,audit[split],flush=True)
    (out/'annotation_audit.json').write_text(json.dumps(audit,indent=2))
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Thebe spatial v3</title>
    <style>body{max-width:1100px;margin:30px auto;font-family:sans-serif;line-height:1.65}img{max-width:100%}</style>
    <h1>Thebe 空间划分与 UNet-L 试跑</h1>
    <p>训练/验证/测试先按空间划分；32 张剖面缓冲。测试集本轮不运行。完整方案：
    <a href="../../docs/THEBE_SPATIAL_V3_TRAINING.md">协议说明</a>。</p>
    <img src="split.png"><img src="patch_cores.png"><h2>固定样本示意</h2><p>示意图只用于检查，不是模型预测。</p>
    <img src="patch_examples.png"><h2>实时验证曲线</h2>
    <p>96 核心快速验证用于观察；完整验证才选择 checkpoint。首轮完成后产生曲线，刷新页面更新。</p>
    <img src="../../runs/thebe_spatial_v3_unet_l_trial/live_curves.png">
    <p><a href="checks.json">不变量检查</a> · <a href="annotation_audit.json">训练/验证标签统计</a> ·
    <a href="../../runs/thebe_spatial_v3_unet_l_trial/train.log">训练日志</a></p>''')

if __name__=='__main__': main()
