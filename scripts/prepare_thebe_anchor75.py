"""Build train-only foreground pools and a replayable 60-epoch sampling plan."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LabelReader:
    def __init__(self, root):
        self.manifest = json.loads((Path(root)/'manifest.json').read_text())
        self.record = self.manifest['records']['train']
        self.shape = np.array(self.record['shape'])
        self.y = np.load(self.record['labels'], mmap_mode='r')
        b = np.load(self.record['support'])
        self.first, self.last = b['first'], b['last']

    def stats(self, origin):
        origin = np.asarray(origin)
        start = np.maximum(origin, 0); end = np.minimum(origin+128, self.shape)
        if np.any(end <= start):
            return 0, 0, 0
        sl = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        z = np.arange(start[0], end[0])[:, None, None]
        valid = (z >= self.first[sl[1:]]) & (z < self.last[sl[1:]])
        pos = (self.y[sl] > 0) & valid
        a = np.maximum(start, origin+32)-start; b = np.minimum(end, origin+96)-start
        core = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
        return int(valid.sum()), int(pos.sum()), int(pos[core].sum())

    def random_origin(self, rng):
        # Support count only: avoid repeatedly loading amplitudes or labels.
        for _ in range(10000):
            o = np.array([rng.integers(n-127) for n in self.shape], dtype=np.int32)
            z,h,w = o
            lo = self.first[h:h+128,w:w+128]; hi = self.last[h:h+128,w:w+128]
            v = np.clip(np.minimum(hi,z+128)-np.maximum(lo,z),0,128).sum()
            if v >= 128**3*.25:
                return o
        raise RuntimeError('Unable to draw supported random window')


def make_anchors(reader, path, seed):
    if path.exists():
        return np.load(path)['anchors']
    rng = np.random.default_rng(seed)
    rows=[]; cell=0
    for z in range(0, reader.shape[0], 128):
        for h in range(0, reader.shape[1], 128):
            for w in range(0, reader.shape[2], 128):
                sl=tuple(slice(int(a),int(min(a+128,n))) for a,n in zip((z,h,w),reader.shape))
                f=np.asarray(reader.y[sl]); zz=np.arange(z,sl[0].stop)[:,None,None]
                v=(zz>=reader.first[sl[1:]])&(zz<reader.last[sl[1:]])
                coords=np.argwhere((f>0)&v)
                if len(coords):
                    picked=coords[rng.choice(len(coords), min(len(coords),128),replace=False)]
                    picked=picked+np.array([z,h,w])
                    rows.append(np.column_stack((picked,np.full(len(picked),cell))))
                cell+=1
        print('anchor scan z',z,flush=True)
    a=np.concatenate(rows).astype(np.int32)
    np.savez_compressed(path,anchors=a)
    return a


def build_pool(reader, anchors, path, seed, pool_size):
    if path.exists():
        return dict(np.load(path))
    rng=np.random.default_rng(seed+101)
    ids, starts, counts=np.unique(anchors[:,3],return_index=True,return_counts=True)
    rows=[]; seen=set(); t0=time.time()
    for attempt in range(500000):
        ci=int(rng.integers(len(ids)))
        p=anchors[starts[ci]+rng.integers(counts[ci]),:3]
        lo=np.maximum(0,p-95); hi=np.minimum(reader.shape-128,p-32)
        o=np.array([rng.integers(a,b+1) if a<=b else int(q-rng.integers(32,96))
                    for a,b,q in zip(lo,hi,p)],dtype=np.int32)
        key=tuple(o)
        if key in seen:continue
        v,n,c=reader.stats(o)
        if v<128**3*.25 or c==0:continue
        rows.append([*o,int(ids[ci]),v,n,c]);seen.add(key)
        if attempt%2000==0:
            print('pool',attempt,'accepted',len(rows),'seconds',round(time.time()-t0),flush=True)
        if len(rows)>=pool_size:break
    else:raise RuntimeError(f'Insufficient diverse candidates: {len(rows)}')
    rows=np.array(rows,dtype=np.int32);density=rows[:,5]/rows[:,4]
    threshold=float(np.quantile(density,.25));low=density<=threshold
    result=dict(regular=rows[~low],sparse=rows[low])
    (path.parent/'density_threshold.json').write_text(json.dumps(dict(
        threshold=threshold,rule='lower quartile of train-only spatially balanced foreground candidates',
        absolute_below_0_005=int((density<.005).sum()),total=len(rows)),indent=2))
    np.savez_compressed(path,**result)
    return result


def grouped_pool(pool):
    cells=np.unique(pool[:,3])
    return {int(c):np.flatnonzero(pool[:,3]==c) for c in cells}


def make_plans(reader,pools,out,epochs,samples,seed):
    groups={k:grouped_pool(v) for k,v in pools.items()}
    summaries=[]
    for epoch in range(1,epochs+1):
        target=out/f'epoch_{epoch:03d}.npz'
        if target.exists():
            summaries.append(json.loads((out/f'epoch_{epoch:03d}.json').read_text()));continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,313]))
        origins=[];kinds=[];stats=[];cells=[]
        for _ in range(samples//8):
            used=set()
            for kind in rng.permutation([0,0,0,0,1,1,2,2]):
                if kind==2:
                    o=reader.random_origin(rng);v,n,c=reader.stats(o);cell=-1
                else:
                    name='regular' if kind==0 else 'sparse';pool=pools[name];g=groups[name]
                    eligible=[x for x in g if x not in used]
                    if not eligible:eligible=list(g)
                    cell=int(rng.choice(eligible));used.add(cell)
                    row=pool[int(rng.choice(g[cell]))];o=row[:3];v,n,c=map(int,row[4:])
                origins.append(o);kinds.append(kind);stats.append((v,n,c));cells.append(cell)
        origins=np.array(origins,dtype=np.int32);kinds=np.array(kinds,dtype=np.uint8);stats=np.array(stats,dtype=np.int32)
        aug=np.random.default_rng(np.random.SeedSequence([seed,epoch,991])).integers(0,2**32,size=samples,dtype=np.uint32)
        for b in kinds.reshape(-1,8):assert np.bincount(b,minlength=3).tolist()==[4,2,2]
        assert np.all(stats[kinds<2,2]>0)
        assert np.all(stats[:,0]>=128**3*.25)
        summary=dict(epoch=epoch,samples=samples,counts=np.bincount(kinds,minlength=3).tolist(),
            positive_windows=int((stats[:,1]>0).sum()),positive_core_windows=int((stats[:,2]>0).sum()),
            empty_windows=int((stats[:,1]==0).sum()),mean_positive_fraction=float(np.mean(stats[:,1]/stats[:,0])),
            unique_origins=len(set(map(tuple,origins))),positive_spatial_cells=len(set(cells)-{-1}),
            padded_windows=int(np.any((origins<0)|(origins+128>reader.shape),axis=1).sum()))
        np.savez_compressed(target,origins=origins,kinds=kinds,counts=stats,cells=np.array(cells),augmentation_seeds=aug)
        (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(summary,indent=2));summaries.append(summary)
        print('plan',epoch,summary['positive_windows'],summary['positive_core_windows'],summary['unique_origins'],flush=True)
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'data/thebe_spatial_v3'))
    p.add_argument('--out',default=str(ROOT/'data/thebe_anchor75_60'))
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--samples',type=int,default=2400)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--pool-size',type=int,default=24000);a=p.parse_args()
    if a.samples%8:raise ValueError('samples must be divisible by 8')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);reader=LabelReader(a.root)
    spec=dict(version='anchor75_quantile_v2',root=str(Path(a.root).resolve()),seed=a.seed,epochs=a.epochs,samples=a.samples,
        pool_size=a.pool_size,protocol_sha256=reader.manifest['protocol_sha256'],
        train_shape=reader.shape.tolist(),support_sha256=sha(reader.record['support']),
        source_sha256=sha(__file__),quota=[4,2,2],density_rule='train_candidate_lower_quartile',core=64,patch=128)
    prep=out/'preparation_spec.json'
    if prep.exists() and json.loads(prep.read_text())!=spec:raise RuntimeError('Preparation spec changed: use a new output directory')
    prep.write_text(json.dumps(spec,indent=2))
    anchors=make_anchors(reader,out/'anchors.npz',a.seed)
    print('anchors',len(anchors),'cells',len(np.unique(anchors[:,3])),flush=True)
    pools=build_pool(reader,anchors,out/'pools.npz',a.seed,a.pool_size)
    summary=make_plans(reader,pools,out,a.epochs,a.samples,a.seed)
    payload=dict(spec=spec,density=json.loads((out/'density_threshold.json').read_text()),pool_cells={k:len(np.unique(v[:,3])) for k,v in pools.items()},
        plans={f'epoch_{i:03d}.npz':sha(out/f'epoch_{i:03d}.npz') for i in range(1,a.epochs+1)},epochs=summary)
    payload['plan_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(payload,indent=2));print('READY',payload['plan_sha256'],flush=True)


if __name__=='__main__':main()
