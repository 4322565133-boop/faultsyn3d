"""Validation-only per-category TP/FP/FN and error-distance audit."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import build
from train.dataset import FaultVolumes,CATEGORIES


def metrics(c):
    tp,fp,fn,near_fp,near_fn=map(int,c)
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),
                recall=tp/max(tp+fn,1),near_fp=near_fp,far_fp=fp-near_fp,
                near_fp_fraction=near_fp/max(fp,1),near_fn=near_fn,far_fn=fn-near_fn)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    ds=FaultVolumes(ROOT/'data/dataset_reproduction_v2','val')
    loader=DataLoader(ds,batch_size=1,num_workers=2,pin_memory=True)
    result={'split':'validation','n_volumes':len(ds),'threshold':.5,
            'distance':'Chebyshev distance <=2 voxels, diagnostic only; no label dilation in IoU/P/R',
            'manifest_sha256':hashlib.sha256((ROOT/'data/dataset_reproduction_v2/manifest.json').read_bytes()).hexdigest(),'runs':{}}
    for run in a.runs:
        path=Path(run);ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
        args=ck['args'];kw=args.get('mkw',{})
        if isinstance(kw,list):
            pairs={}
            for pair in kw:
                k,v=pair.split('=',1)
                for cast in [int,float]:
                    try:v=cast(v);break
                    except ValueError:pass
                pairs[k]=v
            kw=pairs
        model=build(args['model'],**kw).cuda().eval();model.load_state_dict(ck['model'])
        epoch=ck['epoch'];del ck
        sums=torch.zeros(len(CATEGORIES),5,device='cuda',dtype=torch.int64);per_volume=[]
        for i,(x,y,cat) in enumerate(loader):
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x).float().sigmoid()>.5
            truth=y>.5;fp=pred&~truth;fn=~pred&truth
            near_gt=F.max_pool3d(y,5,1,2)>.5
            near_pred=F.max_pool3d(pred.float(),5,1,2)>.5
            counts=torch.stack([(pred&truth).sum(),fp.sum(),fn.sum(),(fp&near_gt).sum(),(fn&near_pred).sum()])
            sums[int(cat[0])]+=counts
            per_volume.append(dict(name=ds.items[i][0],category=ds.items[i][1],**metrics(counts.tolist())))
            if (i+1)%20==0:print(path.name,i+1,flush=True)
        record={'checkpoint_epoch':epoch,'overall':metrics(sums.sum(0).tolist()),
                'categories':{c:metrics(sums[k].tolist()) for k,c in enumerate(CATEGORIES)},'volumes':per_volume}
        result['runs'][str(path)]=record
        dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
        print(path.name,json.dumps(record['categories']),flush=True)
        del model;torch.cuda.empty_cache()


if __name__=='__main__':main()
