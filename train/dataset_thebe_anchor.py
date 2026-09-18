"""Replayable anchor75 training and spatial-v3 validation, optionally dual FOV."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from .dataset_thebe_spatial import SpatialThebe, DEFAULT


def transform(a,k,flips):
    a=np.rot90(a,k,(1,2))
    for axis,flip in zip((1,2),flips):
        if flip:a=np.flip(a,axis)
    return np.ascontiguousarray(a)


def down_context(x,mask,sigma=None):
    """Centered normalized low-pass + 2x2x1 box decimation, preserving masks."""
    if sigma is not None:
        num=gaussian_filter(x*mask,sigma,mode='constant',cval=0)
        den=gaussian_filter(mask,sigma,mode='constant',cval=0)
        x=np.divide(num,den,out=np.zeros_like(num),where=den>1e-6)
        x*=mask
    num=gaussian_filter(x*mask,(.8,.8,0),mode='constant',cval=0)
    den=gaussian_filter(mask,(.8,.8,0),mode='constant',cval=0)
    def reduce(a):return a.reshape(128,2,128,2,128).mean((1,3))
    num=reduce(num);den=reduce(den);valid=reduce(mask)
    y=np.divide(num,den,out=np.zeros_like(num),where=den>1e-6)
    y[valid==0]=0
    return y,valid


class AnchorThebe(SpatialThebe):
    def __init__(self,split,root=DEFAULT,plans=None,context=False,fast=False):
        super().__init__(split,root,fast=fast)
        self.context=bool(context);self.plans=Path(plans) if plans else None
        self.loaded_epoch=None;self.plan=None
        self.plan_manifest=None
        if split=='train':
            if self.plans is None:raise ValueError('Training requires a verified sampling plan')
            self.plan_manifest=json.loads((self.plans/'manifest.json').read_text())
            if self.plan_manifest['spec']['protocol_sha256']!=self.manifest['protocol_sha256']:
                raise ValueError('Sampling plan and data protocol differ')
            self.samples=self.plan_manifest['spec']['samples']

    def load_epoch(self):
        if self.loaded_epoch!=self.epoch:
            file=self.plans/f'epoch_{self.epoch:03d}.npz'
            if hashlib.sha256(file.read_bytes()).hexdigest()!=self.plan_manifest['plans'][file.name]:
                raise RuntimeError('Sampling plan integrity mismatch')
            self.plan=dict(np.load(file));self.loaded_epoch=self.epoch

    def read_context(self,origin):
        # read() has already opened the original arrays. Clip against record shape,
        # not the underlying train file (which also contains excluded sections).
        shape=np.array([256,256,128]);o=np.asarray(origin)-[64,64,0]
        a=np.maximum(o,0);b=np.minimum(o+shape,self.shape)
        src=tuple(slice(int(v),int(w)) for v,w in zip(a,b))
        dst=tuple(slice(int(v-u),int(w-u)) for v,w,u in zip(a,b,o))
        x=np.zeros(tuple(shape),np.float32);m=np.zeros_like(x)
        s=np.asarray(self._x[src]);z=np.arange(a[0],b[0])[:,None,None]
        v=(z>=self.first[src[1:]])&(z<self.last[src[1:]])&np.isfinite(s)
        x[dst]=np.where(v,s,0);m[dst]=v
        return x,m

    def __getitem__(self,i):
        k=0;flips=(False,False);sigma=None
        if self.split=='train':
            self.load_epoch();origin=tuple(map(int,self.plan['origins'][i]))
            rng=np.random.default_rng(int(self.plan['augmentation_seeds'][i]))
            k=int(rng.integers(4));flips=(rng.random()<.5,rng.random()<.5)
            if rng.random()<.35:sigma=float(rng.uniform(.4,1))
        else:origin=tuple(v-32 for v in self.origins[i])
        x,y,m,ml=self.read(origin)
        if self.split=='train':
            kind=int(self.plan['kinds'][i]);v,n,c=map(int,self.plan['counts'][i])
            if int(m.sum())!=v or int((y*m).sum())!=n:
                raise RuntimeError('Actual data no longer match the sampling plan')
            if kind<2 and (y[32:96,32:96,32:96]*m[32:96,32:96,32:96]).sum()==0:
                raise RuntimeError('Foreground plan has an empty core')
        x,y,m,ml=[transform(t,k,flips) for t in (x,y,m,ml)]
        if sigma is not None:x=gaussian_filter(x,sigma).astype(np.float32)
        v=m>0;mean=float(x[v].mean()) if v.any() else 0.;std=float(x[v].std())+1e-6 if v.any() else 1.
        x=(x-mean)/std;x[~v]=0
        inputs=torch.from_numpy(np.ascontiguousarray(x[None]))
        if self.context:
            cx,cm=self.read_context(origin);cx,cm=down_context(cx,cm,sigma)
            cx,cm=[transform(t,k,flips) for t in (cx,cm)]
            cx=(cx-mean)/std;cx[cm==0]=0
            extent=np.array([256,256,128],dtype=np.float32)
            if k%2:extent[[1,2]]=extent[[2,1]]
            inputs=(inputs,torch.from_numpy(cx[None]),torch.from_numpy(cm[None]),torch.from_numpy(extent))
        if self.split!='train':
            core=np.zeros_like(m);core[32:96,32:96,32:96]=1;m*=core;ml*=core
        return inputs,*[torch.from_numpy(np.ascontiguousarray(t[None])) for t in (y,m,ml)]
