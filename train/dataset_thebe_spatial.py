"""Spatially sealed Thebe patches and exhaustive, nonoverlapping evaluation cores."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

DEFAULT = Path(__file__).resolve().parents[1]/'data/thebe_spatial_v3'

def label_blocks(frac, n_blocks, length=3174, patch=128):
    """contiguous labelled ranges along axis 1 for a label fraction: n_blocks blocks of width
    round(length*frac/n_blocks) (>= patch), centred at evenly spaced positions"""
    if frac >= 1: return [(0, length)]
    w=max(patch, int(round(length*frac/n_blocks)))
    return [(int(round((i+.5)*length/n_blocks-w/2)), int(round((i+.5)*length/n_blocks-w/2))+w) for i in range(n_blocks)]


class SpatialThebe(Dataset):
    """returns (x, y, m_valid, m_label): m_label = signal support AND inside a labelled block; for an
    `unlabeled` view the labels are hidden (y = 0, m_label = 0) so no label can leak into training."""
    def __init__(self, split, root=DEFAULT, samples=2400, seed=2026, fast=False, labeled_ranges=None, unlabeled=False, fg_frac=0.0, fg_min=0.005):
        self.manifest=json.loads((Path(root)/'manifest.json').read_text())
        self.labeled_ranges=labeled_ranges; self.unlabeled=unlabeled
        # foreground oversampling (nnU-Net style): a fraction fg_frac of the training draws must contain at least
        # fg_min * (supported voxels) fault voxels; the rest are plain random windows.  Evaluation is unaffected.
        self.fg_frac=float(fg_frac); self.fg_min=float(fg_min)
        self.record=self.manifest['records'][split]
        self.shape=tuple(self.record['shape']); self.split=split
        self.samples=samples; self.seed=seed; self.epoch=0
        b=np.load(self.record['support']); self.first=b['first']; self.last=b['last']
        self._x=self._y=None
        self.origins=[]
        if split!='train':
            for z,y,x in itertools.product(*(range(0,n,64) for n in self.shape)):
                lo=self.first[y:y+64,x:x+64]; hi=self.last[y:y+64,x:x+64]
                if ((hi>lo)&(hi>z)&(lo<min(z+64,self.shape[0]))).any():
                    self.origins.append((z,y,x))
            if fast:
                inds=np.linspace(0,len(self.origins)-1,min(96,len(self.origins)),dtype=int)
                self.origins=[self.origins[i] for i in inds]
    def __len__(self): return self.samples if self.split=='train' else len(self.origins)
    def read(self, origin):
        if self._x is None:
            self._x=np.load(self.record['seismic'],mmap_mode='r')
            self._y=np.load(self.record['labels'],mmap_mode='r')
        out=[np.zeros((128,128,128),dtype=np.float32) for _ in range(4)]
        start=[max(0,v) for v in origin]; end=[min(n,v+128) for n,v in zip(self.shape,origin)]
        src=tuple(slice(a,b) for a,b in zip(start,end))
        dst=tuple(slice(a-v,b-v) for a,b,v in zip(start,end,origin))
        s=np.asarray(self._x[src]); f=np.asarray(self._y[src])
        z=np.arange(start[0],end[0])[:,None,None]
        lo=self.first[src[1:]]; hi=self.last[src[1:]]
        mask=(z>=lo)&(z<hi)&np.isfinite(s)
        out[0][dst]=np.where(mask,s,0); out[1][dst]=f; out[2][dst]=mask
        lab=mask.copy()
        if self.labeled_ranges is not None:                        # labelled blocks along axis 1
            ys=np.arange(start[1],end[1]); inrange=np.zeros(len(ys),bool)
            for a,b in self.labeled_ranges: inrange|=(ys>=a)&(ys<b)
            lab=lab&inrange[None,:,None]
        out[3][dst]=lab
        if self.unlabeled:                                          # hide labels entirely
            out[1][:]=0; out[3][:]=0
        return out
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+self.epoch*100003+i)
        if self.split=='train':
            need_fg=(self.fg_frac>0) and (rng.random()<self.fg_frac) and not self.unlabeled
            best=None
            for attempt in range(1000):
                if self.labeled_ranges is not None and not self.unlabeled and self.labeled_ranges!=[(0,self.shape[1])]:
                    a,b=self.labeled_ranges[int(rng.integers(len(self.labeled_ranges)))]   # patch fully inside a block
                    origin=(int(rng.integers(self.shape[0]-128+1)), int(rng.integers(a,b-128+1)), int(rng.integers(self.shape[2]-128+1)))
                else:
                    origin=tuple(int(rng.integers(n-128+1)) for n in self.shape)
                x,y,m,ml=self.read(origin)
                if m.mean()<.25: continue
                if not need_fg: break
                fg=float((y*ml).sum()/max(ml.sum(),1))                       # fault fraction of the labelled support
                if fg>=self.fg_min: break
                if best is None or fg>best[0]: best=(fg,x,y,m,ml)
                if attempt>=200: x,y,m,ml=best[1:]; break                    # give up: keep the richest window seen
            else: raise RuntimeError('No sufficiently supported training patch after 1000 attempts')
            k=int(rng.integers(4))
            x,y,m,ml=[np.rot90(t,k,(1,2)).copy() for t in (x,y,m,ml)]
            for axis in (1,2):
                if rng.random()<.5: x,y,m,ml=[np.flip(t,axis).copy() for t in (x,y,m,ml)]
            if rng.random()<.35: x=gaussian_filter(x,sigma=rng.uniform(.4,1)).astype(np.float32)
        else:
            origin=tuple(v-32 for v in self.origins[i]); x,y,m,ml=self.read(origin)
        valid=m>0
        if valid.any(): x=(x-x[valid].mean())/(x[valid].std()+1e-6)
        x[~valid]=0
        if self.split!='train':
            core=np.zeros_like(m); core[32:96,32:96,32:96]=1; m*=core; ml*=core
        return tuple(torch.from_numpy(np.ascontiguousarray(t[None])) for t in (x,y,m,ml))
