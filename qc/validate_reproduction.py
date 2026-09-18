"""Read every output array and reconstruct geometry from recorded parameters."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,collections,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from su.geometry import rotation_matrix
from su.topology import tree_from_spec
from su.optimize import objective
from su.reproduction import surface_graph,validate_surfaces,check_control_domains


def check(job):
    root,row,source=job;root=Path(root);name=row['name'];hashes={}
    m=json.loads((root/'metadata'/f'{name}.json').read_text());cfg=m['config']
    assert m['source_sha256']==source,name
    assert m['category']==row['category'] and m['replicate']==row['k'],name
    nx,ny,nz=m['grid'];shape=(nz,ny,nx);arrays={}
    for kind,dtype in [('seismic',np.float32),('labels',np.uint8),('instances',np.uint8),('confidence',np.uint8)]:
        path=root/kind/f'{name}.dat';raw=path.read_bytes()
        assert len(raw)==nx*ny*nz*np.dtype(dtype).itemsize,(name,kind,'size')
        arrays[kind]=np.frombuffer(raw,dtype=dtype).reshape(shape)
        hashes[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
    s,l,inst,conf=[arrays[k] for k in ['seismic','labels','instances','confidence']]
    assert np.isfinite(s).all() and abs(float(s.mean()))<1e-4 and abs(float(s.std())-1)<1e-4,name
    assert np.isin(l,[0,1]).all() and np.array_equal(l>0,inst>0),name
    assert not np.any(conf[l==0]) and abs(float(l.mean())-m['fault_fraction'])<1e-10,name
    assert np.array_equal(np.fromfile(root/'labels_full'/f'{name}.dat',np.uint8).reshape(shape),l),name
    assert set(np.unique(inst))==set(range(m['n_faults']+1)),(name,'lost instance')
    assert m['pso']['penalty']==0 and m['geometry_qc_pass'],name
    tree=tree_from_spec([(p,tuple(e)) for p,e in cfg['tree_specs'][m['category']]])
    z=np.vstack([np.zeros(3),m['zeta']]);assert objective(tree,z,len(tree))==0,(name,'recomputed PSO')
    main=m['main'];centre=np.array(main['centre']);R=rotation_matrix(main['strike_deg'],main['dip_deg'])
    ns=cfg['surface_samples'];xs=np.linspace(-main['half_len'],main['half_len'],ns);ys=np.linspace(-main['half_wid'],main['half_wid'],ns)
    U,V=np.meshgrid(xs,ys,indexing='ij');anchors=(np.array([f['anchors'] for f in m['faults']])-centre)@R.T
    betas=np.array([[f['beta_strike'],f['beta_dip']] for f in m['faults']])
    heights=np.array([surface_graph(a,xs,ys,*b) for a,b in zip(anchors,betas)])
    supports=np.array([(abs(U-v['centre_uv'][0])<=v['half_extent_uv'][0])&(abs(V-v['centre_uv'][1])<=v['half_extent_uv'][1]) for v in m['finite_support']])
    valid,inside,edges=validate_surfaces(tree,heights,xs,ys,centre,R,m['grid'],z,supports)
    hull_tests=check_control_domains(tree,anchors,betas,valid,xs,ys)
    assert hull_tests==m['control_hull_tests'],name
    areas=[];max_mesh_error=0.
    path=root/'surfaces'/f'{name}.npz'
    with np.load(path) as saved:
        assert len(saved.files)==m['n_faults'],name
        for i,h in enumerate(heights):
            pts=np.stack([U,V,h],axis=-1)@R+centre
            p=saved[f'f{i}'];assert np.array_equal(np.isfinite(p).all(-1),valid[i]),(name,'mask')
            err=float(np.max(abs(p[valid[i]]-pts[valid[i]])));max_mesh_error=max(err,max_mesh_error)
            assert err<3e-5,(name,'surface export',err)
            gu,gv=np.gradient(h,xs,ys);area=float((np.sqrt(1+gu*gu+gv*gv)*valid[i]*inside[i]).sum()*(xs[1]-xs[0])*(ys[1]-ys[0]))
            assert np.isclose(area,m['faults'][i]['area']),name
            areas.append(area)
            f=m['faults'][i];F=rotation_matrix(f['strike'],f['dip']);points=np.array(f['anchors'])
            assert np.max(abs((points-points.mean(0))@F[2]))<1e-7,name
    for i,f in enumerate(m['faults']):
        assert np.isclose(f['d_max'],main['d_max']*np.sqrt(areas[i]/areas[0])),name
    for path in [path,root/'metadata'/f'{name}.json']:
        hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(name=name,category=m['category'],fault_percent=100*float(l.mean()),
                proxy_visible_fraction=float((conf[l>0]==255).mean()),mesh_max_error=max_mesh_error,
                instance_counts=np.bincount(inst.ravel(),minlength=m['n_faults']+1)[1:].tolist(),
                hashes=hashes,control_hull_tests=hull_tests)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    root=a.root.resolve();manifest=json.loads((root/'manifest.json').read_text());run=json.loads((root/'run.json').read_text())
    source_root=Path(__file__).resolve().parents[1]
    assert manifest['source_sha256']==run['source_sha256']
    assert all(hashlib.sha256((source_root/f).read_bytes()).hexdigest()==h for f,h in run['source_files'].items()), 'source version changed'
    rows=manifest['volumes'];assert len(rows)==manifest['n']==5*run['config']['per_category']
    assert len({v['name'] for v in rows})==len(rows)
    n=run['config']['per_category']
    assert all(sum(v['category']==c for v in rows)==n for c in run['config']['tree_specs'])
    assert all(v['split']==('train' if v['k']<int(.8*n) else 'val' if v['k']<int(.9*n) else 'test') for v in rows)
    expected={v['name'] for v in rows}
    for folder,suffix in [('seismic','.dat'),('labels','.dat'),('labels_full','.dat'),('instances','.dat'),('confidence','.dat'),('metadata','.json'),('surfaces','.npz')]:
        assert {p.stem for p in (root/folder).glob('*'+suffix)}==expected,folder
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i,r in enumerate(ex.map(check,[(str(root),row,manifest['source_sha256']) for row in rows])):
            results.append(r)
            if (i+1)%50==0:print(f'checked {i+1}/{len(rows)}',flush=True)
    sums={p:h for r in results for p,h in r.pop('hashes').items()}
    for filename in ['run.json','manifest.json','summary.json','generation_errors.json']:
        sums[filename]=hashlib.sha256((root/filename).read_bytes()).hexdigest()
    (root/'checksums.sha256').write_text(''.join(f'{sums[p]}  {p}\n' for p in sorted(sums)))
    report=dict(status='PASS',n=len(rows),categories=dict(collections.Counter(v['category'] for v in rows)),splits=dict(collections.Counter(v['split'] for v in rows)),
                fault_percent_mean=float(np.mean([r['fault_percent'] for r in results])),
                mesh_max_error=max(r['mesh_max_error'] for r in results),source_sha256=manifest['source_sha256'],
                tests=['array sizes/dtypes/finite/normalisation','binary/full/instance labels consistent','all instances present','recomputed zero PSO objective','reconstructed clipped graph topology','control triangle domains','exported meshes match reconstruction','area and slip scaling',f'{len(sums)} file checksums'],
                limitations=['Topology is checked on a finite 193x193 chart, with documented contact tolerance.','Visibility proxy is isolated vertical slip divided by a quarter-period; it is not a measured seismic detection rate.'],volumes=results)
    (root/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='volumes'},indent=2))
if __name__=='__main__':main()
