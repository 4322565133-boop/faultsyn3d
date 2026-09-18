"""Small-volume invariants: coverage, masks, no label-driven tiling, deterministic draws."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from train.dataset_thebe_spatial import SpatialThebe
from train.train_thebe_spatial_ddp import loss_fn

def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); shape=(145,139,133)
        x=np.random.default_rng(7).normal(size=shape).astype(np.float32)
        lo=np.full(shape[1:],5,dtype=np.int16); hi=np.full(shape[1:],140,dtype=np.int16)
        lo[0,:]=145; hi[0,:]=0
        valid=(np.arange(shape[0])[:,None,None]>=lo)&(np.arange(shape[0])[:,None,None]<hi)
        x[~valid]=0; x[64,64,64]=0  # a true interior zero must remain valid
        y=(x>1).astype(np.uint8)
        np.save(root/'x.npy',x); np.save(root/'y.npy',y); np.savez(root/'bounds.npz',first=lo,last=hi)
        record=dict(shape=shape,seismic=str(root/'x.npy'),labels=str(root/'y.npy'),support=str(root/'bounds.npz'))
        (root/'manifest.json').write_text(json.dumps(dict(records={'train':record,'val':record})))
        ds=SpatialThebe('val',root); cover=np.zeros(shape,np.uint8); nvalid=0
        for i,(z,y0,x0) in enumerate(ds.origins):
            _,_,m=ds[i]; core=m.numpy()[0,32:96,32:96,32:96]
            stop=tuple(min(v+64,n) for v,n in zip((z,y0,x0),shape))
            sl=tuple(slice(v,w) for v,w in zip((z,y0,x0),stop))
            sz=tuple(slice(0,w-v) for v,w in zip((z,y0,x0),stop))
            cover[sl]+=core[sz].astype(np.uint8); nvalid+=int(core.sum())
        assert np.array_equal(cover,valid.astype(np.uint8)), 'Gap or double counting at final/padded cores'
        assert nvalid==int(valid.sum())
        assert ds.read((0,0,0))[2][64,64,64]==1
        original=ds.origins.copy(); np.save(root/'y.npy',np.zeros(shape,np.uint8))
        assert SpatialThebe('val',root).origins==original, 'Tiling depends on labels'
        tr=SpatialThebe('train',root,samples=8); one=tr[2]; two=tr[2]
        assert all(torch.equal(a,b) for a,b in zip(one,two))
        tr.epoch=1; assert not torch.equal(one[0],tr[2][0])
        logits=torch.randn(1,1,8,8,8,requires_grad=True); target=torch.zeros_like(logits); mask=torch.zeros_like(logits)
        mask[:,:,2:6,2:6,2:6]=1; loss=loss_fn(logits,target,mask); loss.backward()
        assert torch.isfinite(logits.grad).all() and (logits.grad[mask==0]==0).all()
        changed=logits.detach().clone(); changed[mask==0]=100
        assert torch.allclose(loss.detach(),loss_fn(changed,target,mask))
    real=SpatialThebe('train'); val=SpatialThebe('val'); fast=SpatialThebe('val',fast=True)
    intervals=real.manifest['intervals_zero_based_half_open']
    assert intervals['train'][1]+32==intervals['val'][0]
    assert intervals['val'][1]+32==intervals['test'][0]
    result=dict(status='PASS',coverage='every supported voxel exactly once including tails',
        masking='interior zero retained; padding ignored by loss',label_independent_validation=True,
        deterministic_training=True,buffer_sections=32,full_val_cores=len(val),fast_val_cores=len(fast))
    out=ROOT/'reports/thebe_spatial_v3'; out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
