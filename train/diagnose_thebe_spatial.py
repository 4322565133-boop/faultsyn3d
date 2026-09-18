"""Validation-only diagnosis of existing checkpoints. Never opens test data.

Reports exact threshold-grid counts, approximate AP, per-core errors and boundary strata.
No training, label modification, or automatic change to the selected checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from models import build
from train.dataset_thebe_spatial import SpatialThebe, DEFAULT

def summarize_counts(tp,fp,fn):
    return dict(iou=tp/max(tp+fp+fn,1),dice=2*tp/max(2*tp+fp+fn,1),
        precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))

def threshold_counts(p,y,thresholds):
    # right=False puts p==threshold below the cut: matches strict p > threshold.
    ix=torch.bucketize(p.contiguous(),thresholds,right=False)
    pos=torch.bincount(ix[y],minlength=len(thresholds)+1)
    neg=torch.bincount(ix[~y],minlength=len(thresholds)+1)
    tp=pos.flip(0).cumsum(0).flip(0)[1:]
    fp=neg.flip(0).cumsum(0).flip(0)[1:]
    return torch.stack((tp,fp,pos.sum()-tp),1)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--checkpoint',choices=['best_fullval','last'],default='best_fullval')
    parser.add_argument('--root',default=str(DEFAULT)); parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--fast',action='store_true')
    parser.add_argument('--out',default=str(ROOT/'reports/thebe_protocol_review_20260918/validation_diagnostics'))
    args=parser.parse_args(); device=torch.device(f'cuda:{args.gpu}'); torch.cuda.set_device(device)
    ds=SpatialThebe('val',args.root,fast=args.fast)
    loader=DataLoader(ds,batch_size=1,num_workers=args.workers,pin_memory=True)
    ts=torch.arange(1,20,device=device,dtype=torch.float32)/20
    dest=Path(args.out); dest.mkdir(parents=True,exist_ok=True)
    for run in args.runs:
        run=Path(run); cfg=json.loads((run/'config.json').read_text())
        ck=torch.load(run/f'{args.checkpoint}.pt',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==ds.manifest['protocol_sha256']
        model=build(cfg['model'],**cfg['model_kwargs']).to(device)
        model.load_state_dict(ck['model']); model.eval()
        total=torch.zeros((len(ts),3),dtype=torch.int64,device=device)
        hp=torch.zeros(2001,dtype=torch.int64,device=device); hn=hp.clone()
        rows=[]; t0=time.time()
        for i,batch in enumerate(loader):
            x,y,m=batch[:3]  # SpatialThebe now also returns the labelled-support mask.
            x,y,m=[t.to(device,non_blocking=True) for t in (x,y,m)]
            with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
            p=logits.float().sigmoid()[m>.5]; gt=y[m>.5]>.5
            counts=threshold_counts(p,gt,ts); total+=counts
            bins=(p*2000).floor().long().clamp(0,2000)
            hp+=torch.bincount(bins[gt],minlength=2001); hn+=torch.bincount(bins[~gt],minlength=2001)
            tp,fp,fn=counts[9].tolist(); z,y0,x0=ds.origins[i]
            padded=any(v<32 or v+96>n for v,n in zip((z,y0,x0),ds.shape))
            rows.append(dict(index=i,z=z,y=y0,x=x0,padded_context=int(padded),valid=int(len(p)),
                positive=int(gt.sum()),tp=tp,fp=fp,fn=fn,**summarize_counts(tp,fp,fn)))
            if i%300==0: print(run.name,args.checkpoint,i,len(ds),round(time.time()-t0),flush=True)
        counts=total.cpu().numpy(); grid=[dict(threshold=float(t),**summarize_counts(*map(int,c))) for t,c in zip(ts.cpu(),counts)]
        cp=np.cumsum(hp.cpu().numpy()[::-1]); cn=np.cumsum(hn.cpu().numpy()[::-1]); recall=cp/max(cp[-1],1)
        ap=float(np.sum(np.diff(np.r_[0.,recall])*cp/np.maximum(cp+cn,1)))
        groups={}
        for name,predicate in [('padded_context',lambda r:r['padded_context']),('interior_context',lambda r:not r['padded_context']),
            ('empty_label_core',lambda r:r['positive']==0),('positive_label_core',lambda r:r['positive']>0)]:
            group=[r for r in rows if predicate(r)]
            sums={k:sum(r[k] for r in group) for k in ('tp','fp','fn','valid','positive')}
            groups[name]=dict(cores=len(group),**sums,**summarize_counts(sums['tp'],sums['fp'],sums['fn']),
                false_positive_voxel_rate=sums['fp']/max(sums['valid']-sums['positive'],1))
        scored=sum(r['valid'] for r in rows)
        if not args.fast: assert scored==ds.record['support_voxels']
        result=dict(run=str(run),checkpoint=args.checkpoint,epoch=ck['epoch'],split='val',fast=args.fast,
            protocol_sha256=ds.manifest['protocol_sha256'],scored_voxels=scored,at_0_5=grid[9],
            val_selected_threshold=max(grid,key=lambda r:r['iou']),threshold_grid=grid,histogram_ap=ap,
            strata=groups,seconds=time.time()-t0,
            note='Best threshold on validation is diagnostic; lock before any test evaluation. Fast subset must not select threshold.')
        stem=f'{run.name}_{args.checkpoint}_'+('fast' if args.fast else 'full')
        (dest/f'{stem}.json').write_text(json.dumps(result,indent=2))
        with open(dest/f'{stem}_cores.csv','w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({k:result[k] for k in ('run','epoch','at_0_5','val_selected_threshold','histogram_ap')}),flush=True)
        del model,ck; torch.cuda.empty_cache()

if __name__=='__main__': main()
