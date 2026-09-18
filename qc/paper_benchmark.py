"""Thirty independent PSO runs per Fig. 12 tree, using the paper's domain."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/faultsyn_mpl')
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from su.topology import tree_from_spec
from su.optimize import solve_pso,axis_bounds,axis_origins
from su.geometry import rotation_matrix,reference_plane_corners

out=Path(__file__).resolve().parent/'out/reproduction_v2_benchmark'
out.mkdir(parents=True,exist_ok=True)
centre=np.array([128.,128.,64.]);R=rotation_matrix(90.,90.)
origins=axis_origins(centre,R,reference_plane_corners(centre,R,128.,64.))
lo,hi=axis_bounds((256,256,128),centre,R,origins,4.)
specs={'tree1':[0,0],'tree2':[0,0,0],'tree3':[0,1,2],'tree4':[0,0,0,1,2,3]}
report={}
for name,parents in specs.items():
    tree=tree_from_spec([(p,('Y','Y')) for p in parents]);runs=[]
    for seed in range(30):
        t=time.monotonic();z,info=solve_pso(tree,lo,hi,np.random.default_rng(41000+seed))
        runs.append(dict(seed=41000+seed,seconds=time.monotonic()-t,zeta=z.tolist(),**info))
    report[name]=dict(parents=parents,runs=runs,converged=sum(r['penalty']==0 for r in runs),
                      median_iterations=float(np.median([r['iterations'] for r in runs])),
                      max_iterations=max(r['iterations'] for r in runs))
    print(name,report[name]['converged'],report[name]['median_iterations'],report[name]['max_iterations'],flush=True)
(out/'benchmark.json').write_text(json.dumps(report,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,(name,result) in zip(axes.ravel(),report.items()):
    for run in result['runs']:
        ax.plot(run['history'],alpha=.3,lw=1,color='#246a91' if run['penalty']==0 else '#b7473a')
    ax.set(title=f"{name}: {result['converged']}/30 reach zero",xlabel='PSO update',ylabel='Penalty')
fig.tight_layout();fig.savefig(out/'convergence.png',dpi=160)
