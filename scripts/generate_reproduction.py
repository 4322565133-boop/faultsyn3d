"""Generate the audited reproduction dataset with resumable, atomic outputs."""
from __future__ import annotations
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from su.model import MainFault
from su.topology import tree_from_spec
from su.reproduction import generate,RejectedSample


def sample_main(cat,cfg,rng):
    r=cfg['main_fault_ranges'][cat];nx,ny,nz=cfg['grid']
    return MainFault(
        strike_deg=float(rng.uniform(*r['strike'])),dip_deg=float(rng.uniform(*r['dip'])),
        centre=(nx/2+float(rng.uniform(-10,10)),ny/2+float(rng.uniform(-10,10)),nz/2+float(rng.uniform(-8,8))),
        half_len=float(rng.uniform(*cfg['main_half_extent_ratio'])*nx),
        half_wid=float(rng.uniform(*cfg['main_half_extent_ratio'])*nz),
        d_max=float(rng.uniform(*cfg['main_d_max'])),phi_dis_deg=float(rng.uniform(*r['phi_dis'])),
        throughgoing=r['throughgoing'],drag_mode=r['drag'])


def atomic_json(path,data):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    temp.replace(path)


def worker(job):
    cfg,out,cat,k,index,source_hash=job
    out=Path(out);name=f'{index:06d}_{cat}';start=time.monotonic()
    failures=collections.Counter()
    cat_id=list(cfg['tree_specs']).index(cat)
    for attempt in range(cfg['max_attempts']):
        # Independent streams for each sample AND each attempt, independent
        # of worker count, completion order and failures in other samples.
        seed_sequence=np.random.SeedSequence([cfg['seed'],cat_id,k,attempt])
        rng=np.random.default_rng(seed_sequence)
        tree=tree_from_spec([(p,tuple(e)) for p,e in cfg['tree_specs'][cat]])
        main=sample_main(cat,cfg,rng)
        try:
            result=generate(tree,main,cfg,rng,cat)
        except RejectedSample as e:
            failures[str(e)]+=1
            continue
        break
    else:
        return dict(name=name,error='attempt_limit',failures=dict(failures))
    for key,kind,dtype in [('seismic','seismic',np.float32),('label','labels',np.uint8),
                           ('instances','instances',np.uint8),('confidence','confidence',np.uint8)]:
        path=out/kind/f'{name}.dat';tmp=path.with_suffix('.dat.tmp')
        result[key].astype(dtype).tofile(tmp);tmp.replace(path)
    # Full labels are identical by construction; a relative link preserves
    # the old directory convention without duplicating 2 GB of identical data.
    link=out/'labels_full'/f'{name}.dat'
    if not link.exists():link.symlink_to(Path('../labels')/f'{name}.dat')
    path=out/'surfaces'/f'{name}.npz';tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**{f'f{i}':p for i,p in result['surfaces'].items()})
    tmp.replace(path)
    meta=result['meta']
    meta.update(index=index,replicate=k,seed_entropy=[cfg['seed'],cat_id,k,attempt],
                accepted_attempt=attempt+1,rejections=dict(failures),
                seconds=time.monotonic()-start,source_sha256=source_hash)
    atomic_json(out/'metadata'/f'{name}.json',meta)
    return dict(name=name,percent=100*meta['fault_fraction'],attempts=attempt+1,
                visible=meta['label_visible_fraction'],seconds=meta['seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/reproduction_v2.json')
    p.add_argument('--out',type=Path,default=ROOT/'data/dataset_reproduction_v2')
    p.add_argument('--per-category',type=int)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.per_category is not None:cfg['per_category']=a.per_category
    cats=list(cfg['tree_specs']);n=cfg['per_category'];out=a.out.resolve()
    for kind in ['seismic','labels','labels_full','metadata','surfaces','instances','confidence']:
        (out/kind).mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__)]+[ROOT/'su'/f'{m}.py' for m in ['reproduction','geometry','optimize','topology','displacement','stratigraphy','model']]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    source_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    run=dict(config=cfg,source_sha256=source_hash,source_files=hashes,
             numpy=np.__version__,python=sys.version)
    if (out/'run.json').exists():
        previous=json.loads((out/'run.json').read_text())
        if previous['config']!=cfg or previous['source_sha256']!=source_hash:
            raise SystemExit('Refusing to mix configurations or source versions; use a new output directory.')
    else:atomic_json(out/'run.json',run)
    jobs=[];manifest=[]
    for ci,cat in enumerate(cats):
        for k in range(n):
            index=ci*n+k;name=f'{index:06d}_{cat}'
            split='train' if k<int(.8*n) else 'val' if k<int(.9*n) else 'test'
            manifest.append(dict(name=name,category=cat,k=k,index=index,split=split))
            path=out/'metadata'/f'{name}.json'
            if not path.exists():jobs.append((cfg,str(out),cat,k,index,source_hash))
    jobs.sort(key=lambda job: job[3])  # Interleave categories to balance expensive trees.
    print(f'target={len(manifest)} existing={len(manifest)-len(jobs)} workers={a.workers}',flush=True)
    errors=[];completed=len(manifest)-len(jobs)
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(worker,j) for j in jobs]
        for future in as_completed(futures):
            result=future.result()
            if 'error' in result:
                errors.append(result);print('FAILED '+json.dumps(result),flush=True)
            else:
                completed+=1
                print(f"{completed}/{len(manifest)} {result['name']} fault={result['percent']:.3f}% visible={result['visible']:.3f} attempts={result['attempts']} seconds={result['seconds']:.1f}",flush=True)
    atomic_json(out/'generation_errors.json',errors)
    if errors:raise SystemExit(f'{len(errors)} samples failed; no completed manifest written')
    atomic_json(out/'manifest.json',dict(n=len(manifest),grid=cfg['grid'],volumes=manifest,
                                        source_sha256=source_hash))
    summary={};all_failures=collections.Counter()
    for cat in cats:
        meta=[json.loads((out/'metadata'/f"{m['name']}.json").read_text()) for m in manifest if m['category']==cat]
        for m in meta:all_failures.update(m['rejections'])
        summary[cat]=dict(n=len(meta),fault_percent_mean=float(np.mean([m['fault_fraction']*100 for m in meta])),
                         fault_percent_range=[float(fn([m['fault_fraction']*100 for m in meta])) for fn in [np.min,np.max]],
                         visible_fraction_mean=float(np.mean([m['label_visible_fraction'] for m in meta])),
                         attempts_mean=float(np.mean([m['accepted_attempt'] for m in meta])),
                         pso_zero_count=sum(m['pso']['penalty']==0 for m in meta),
                         geometry_qc_pass_count=sum(m['geometry_qc_pass'] for m in meta))
    atomic_json(out/'summary.json',dict(categories=summary,rejections=dict(all_failures),n=len(manifest)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
