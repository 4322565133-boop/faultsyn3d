"""Versioned, signal-only spatial protocol. No held-out labels select samples/ROI."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/thebe_spatial_v3'

def bounds(path, section_stop):
    x = np.load(path, mmap_mode='r')
    first = np.full((x.shape[1], section_stop), x.shape[0], np.int16)
    last = np.zeros_like(first)
    for z in range(0, x.shape[0], 32):
        active = np.isfinite(x[z:z+32, :, :section_stop]) & (x[z:z+32, :, :section_stop] != 0)
        any_ = active.any(0)
        lo = z + active.argmax(0)
        hi = z + len(active) - active[::-1].argmax(0)
        first = np.where(any_, np.minimum(first, lo), first)
        last = np.where(any_, np.maximum(last, hi), last)
    return first, last

def prepare(split, count, stop):
    if split == 'train':
        xp = ROOT/'data/thebe_cubes/train_seis.npy'
        yp = ROOT/'data/thebe_cubes/train_fault.npy'
    else:
        xp, yp = OUT/f'{split}_seis.npy', OUT/f'{split}_fault.npy'
        ready = OUT/f'{split}_cache.ready'
        if not ready.exists():
            offset = 0
            shape = (1537, 3174, stop)
            xx = np.lib.format.open_memmap(xp, mode='w+', dtype='float32', shape=shape)
            yy = np.lib.format.open_memmap(yp, mode='w+', dtype='uint8', shape=shape)
            for i in range(1, count+1):
                if offset >= stop: break
                for kind, target in [('seis', xx), ('fault', yy)]:
                    path = ROOT/f'data/thebe/{kind}/{kind}{split}{i}.npz'
                    print(f'loading {path.name}', flush=True)
                    with np.load(path) as f:
                        a = f['arr_0']
                        n = min(len(a), stop-offset)
                        target[:, :, offset:offset+n] = a[:n].transpose(2, 1, 0)
                        del a
                offset += n
            assert offset == stop
            xx.flush(); yy.flush()
            ready.write_text('complete\n')
    bp = OUT/f'{split}_support.npz'
    if not bp.exists():
        print(f'signal-only support: {split}', flush=True)
        lo, hi = bounds(xp, stop)
        np.savez(bp, first=lo, last=hi)
    b = np.load(bp)
    stat = dict(valid_traces=int((b['last'] > b['first']).sum()),
                support_voxels=int(np.maximum(b['last']-b['first'], 0).astype(np.int64).sum()))
    return dict(seismic=str(xp), labels=str(yp), support=str(bp), shape=[1537,3174,stop], **stat)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--with-test', action='store_true'); a=p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records={s:prepare(s,n,k) for s,n,k in [('train',9,868),('val',2,168)]}
    if a.with_test: records['test']=prepare('test',7,703)
    spec=dict(version='thebe_spatial_v3', axes=['sample','raw_axis_1','raw_axis_0'],
        intervals_zero_based_half_open={'train':[0,868],'val':[900,1068],'test':[1100,1803]},
        excluded_sections=[[868,900],[1068,1100]],
        support='per trace first-to-last finite nonzero seismic sample; interior zero amplitudes are valid',
        patch=128, evaluation_core=64, halo=32, fast_val_cores=96,
        train_sampling='uniform origins over training extent; accept >=25% signal support; no label filtering',
        selection='full validation core-stitched IoU at threshold 0.5 every 10 epochs and final',
        test_policy='sealed until model/protocol frozen; not run by training entry',
        records=records)
    # Protocol hash is independent of whether sealed test cache has been materialized.
    stable={k:v for k,v in spec.items() if k!='records'}
    spec['protocol_sha256']=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    for s,r in records.items():
        r['support_sha256']=hashlib.sha256(Path(r['support']).read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(spec,indent=2))
    print(json.dumps(spec,indent=2),flush=True)

if __name__=='__main__': main()
