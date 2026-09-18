"""Tree and orthogonal sections through exported, clipped fault surfaces."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/faultsyn_mpl')
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from su.geometry import rotation_matrix
from qc.render3d import FAULT_COLORS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--picked',type=Path,required=True);a=p.parse_args()
    names=a.picked.read_text().split();fig,axes=plt.subplots(len(names),3,figsize=(13,3*len(names)),squeeze=False)
    for name,axs in zip(names,axes):
        m=json.loads((a.root/'metadata'/f'{name}.json').read_text());main=m['main'];R=rotation_matrix(main['strike_deg'],main['dip_deg'])
        ns=m['config']['surface_samples'];locals_=[]
        with np.load(a.root/'surfaces'/f'{name}.npz') as saved:
            for i in range(m['n_faults']):
                P=saved[f'f{i}'];inside=np.isfinite(P).all(-1)&np.all((P>=0)&(P<=127),axis=-1)
                q=(P-main['centre'])@R.T;q[~inside]=np.nan;locals_.append(q)
        positions={0:(.5,0)};levels={0:0}
        for node in m['tree'][1:]:levels[node['id']]=levels[node['parent']]+1
        for lev in sorted(set(levels.values())):
            ids=[i for i,l in levels.items() if l==lev]
            for x,i in zip(np.linspace(.15,.85,len(ids)) if len(ids)>1 else [.5],ids):positions[i]=(x,-lev)
        ax=axs[0]
        for node in m['tree'][1:]:
            i,pid=node['id'],node['parent'];x,y=positions[pid];xx,yy=positions[i]
            ax.annotate('',(xx,yy),(x,y),arrowprops=dict(arrowstyle='->',color='.35'))
            ax.text((x+xx)/2,(y+yy)/2,','.join(node['in_edge']),fontsize=8,backgroundcolor='white',ha='center')
        for i,(x,y) in positions.items():ax.scatter(x,y,s=330,c=FAULT_COLORS[i],zorder=3);ax.text(x,y,str(i),ha='center',va='center',zorder=4)
        ax.set_xlim(0,1);ax.set_ylim(-max(levels.values())-.5,.5);ax.axis('off');ax.set_title(name,fontsize=10)
        chosen=[]
        for direction,ax in enumerate(axs[1:]):
            # Strike: variable X at fixed Y. Dip: variable Y at fixed X.
            counts=np.array([np.isfinite(q[...,0]).sum(axis=0 if direction==0 else 1) for q in locals_])
            score=(counts>=8).sum(0)*10000+np.minimum(counts,30).sum(0)-.01*abs(np.arange(ns)-ns//2)
            index=int(score.argmax());chosen.append(index)
            for i,q in enumerate(locals_):
                trace=q[:,index] if direction==0 else q[index,:]
                ax.plot(trace[:,direction],trace[:,2],color=FAULT_COLORS[i],lw=1.8,label=f'Fault {i}')
            ax.set_xlabel('Local X (strike)' if direction==0 else 'Local Y (dip)');ax.set_ylabel('Local Z (normal)')
            ax.set_title(('Strike section' if direction==0 else 'Dip section')+f' | chart index {index}',fontsize=10);ax.grid(alpha=.2)
            ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Recorded tree and sections of the exported finite surfaces\nSections maximise the number of intersected faults; no geometry is moved for display.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(a.out,dpi=160);plt.close(fig);print(a.out)
if __name__=='__main__':main()
