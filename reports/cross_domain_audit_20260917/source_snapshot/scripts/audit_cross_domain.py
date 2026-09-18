"""Frozen existing-checkpoint audit; no training or target threshold selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,json,hashlib,argparse,time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter,gaussian_filter1d
from scipy.signal import hilbert
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train.analyze_loss_vs_iou import load
from train.dataset import FaultVolumes,CATEGORIES
from train.dataset_wu import WuXYC
from train.eval_cross import FaultSeg3DVal
ROOT=Path('reports/cross_domain_audit_20260917')
RUNS={'unet':'runs/maxvit3d_unet_l_ddp','maxvit':'runs/maxvit3d_v2_oldrecipe_all3','hybrid':'runs/maxvit3d_unet_lg_ddp'}
STYLES=['noise_0.25','noise_0.50','lowpass_0.6','lowpass_1.0','phase_30','phase_60']
NB=2000

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def counts(p,y):
 y=y.astype(bool);pr=p>.5;yd=maximum_filter(y,size=5,mode='constant');pd=maximum_filter(pr,size=5,mode='constant')
 out={k:int(v) for k,v in dict(tp=(pr&y).sum(),fp=(pr&~y).sum(),fn=(~pr&y).sum(),matched_pred=(pr&yd).sum(),matched_gt=(y&pd).sum(),positive=y.sum(),predicted=pr.sum(),voxels=y.size,confident_fn=((p<.05)&y).sum()).items()}
 b=np.rint(p*(NB-1)).astype(np.int32)
 hp=np.bincount(b[y],minlength=NB);hn=np.bincount(b[~y],minlength=NB)
 return out,hp,hn

def metrics(c,hp=None,hn=None):
 tp,fp,fn=c['tp'],c['fp'],c['fn']
 m=dict(iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),dice=2*tp/max(2*tp+fp+fn,1),tol2_precision=c['matched_pred']/max(c['predicted'],1),tol2_recall=c['matched_gt']/max(c['positive'],1),label_fraction=c['positive']/c['voxels'],predicted_fraction=c['predicted']/c['voxels'])
 if hp is not None:
  cp=np.cumsum(hp[::-1]);cn=np.cumsum(hn[::-1]);r=cp/max(hp.sum(),1);pr=cp/np.maximum(cp+cn,1)
  m['ap']=float(np.sum(np.diff(np.r_[0.,r])*pr))
 return m

def transform(x,style,name):
 kind,v=style.split('_');v=float(v)
 if kind=='noise':
  seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
  z=x+np.random.default_rng(seed).normal(size=x.shape).astype(np.float32)*v
 elif kind=='lowpass':z=gaussian_filter1d(x,v,axis=0,mode='reflect')
 else:
  pad=32;t=np.pad(x,((pad,pad),(0,0),(0,0)),mode='reflect');a=hilbert(t,axis=0)
  z=(np.cos(np.deg2rad(v))*t-np.sin(np.deg2rad(v))*a.imag)[pad:-pad]
 return ((z-z.mean())/(z.std()+1e-6)).astype(np.float32)

def metadata(name):
 m=json.loads((Path('data/dataset_reproduction_v2/metadata')/(name+'.json')).read_text())
 fs=m['faults']
 return dict(dip=float(m['main']['dip_deg']),max_slip=float(m['main']['d_max']),n_faults=int(m['n_faults']),tree_depth=int(m['tree_depth']),curvature=float(np.mean([abs(f.get('beta_dip',0))+abs(f.get('beta_strike',0)) for f in fs])),snr_db=float(m['snr_db']),freq_hz=float(m['freq_hz']),visible_proxy=float(m['label_visible_fraction']))

def save_json(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def main():
 a=argparse.ArgumentParser();a.add_argument('--model',choices=RUNS,required=True);a.add_argument('--gpu',type=int,default=0);a.add_argument('--smoke',action='store_true');a=a.parse_args()
 torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
 dest=ROOT/(a.model+('_smoke' if a.smoke else ''));dest.mkdir(parents=True,exist_ok=True)
 run=RUNS[a.model];device=torch.device(f'cuda:{a.gpu}');m,ep=load(run,device)
 ck=torch.load(Path(run)/'best.pt',map_location='cpu',weights_only=False)
 header=dict(model=a.model,run=run,checkpoint='best.pt',checkpoint_sha256=sha(Path(run)/'best.pt'),epoch=ep,selection=ck['args'].get('select','historical_val_loss'),args=ck['args'],params=sum(p.numel() for p in m.parameters()),torch=torch.__version__,gpu=torch.cuda.get_device_name(device),threshold=.5,normalization='per-volume z-score',ap='2000-bin histogram approximation',tolerance='Chebyshev <=2 voxels; separate predicted and GT matches',script_sha256=sha(__file__))
 save_json(dest/'provenance.json',header);del ck
 sets={'v2_val':FaultVolumes('data/dataset_reproduction_v2','val'),'v2_test':FaultVolumes('data/dataset_reproduction_v2','test'),'faultseg3d20':FaultSeg3DVal(),'wu_style_test':WuXYC('test')}
 subset={nm for cat in CATEGORIES for nm,c in sets['v2_val'].items if c==cat and nm in [n for n,k in sets['v2_val'].items if k==cat][:5]}
 tasks=[(k,ds,'clean') for k,ds in sets.items()]+[(f'v2_style_{s}',sets['v2_val'],s) for s in STYLES]
 for key,ds,style in tasks:
  rows=[];hp=np.zeros(NB,dtype=np.int64);hn=hp.copy();summed={};start=time.time()
  for i in range(len(ds)):
   if key.startswith('v2'):name,cat=ds.items[i]
   elif key=='faultseg3d20':name=f'wu_public_{ds.ids[i]:02d}';cat='public'
   else:name,cat=ds.ds.items[i]
   if style!='clean' and name not in subset:continue
   x,y,_=ds[i];x=x.numpy()[0];y=y.numpy()[0]>.5
   if style!='clean':x=transform(x,style,name)
   assert np.isfinite(x).all()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    p=m(torch.from_numpy(np.ascontiguousarray(x))[None,None].to(device)).float().sigmoid()[0,0].cpu().numpy()
   assert np.isfinite(p).all()
   c,h1,h0=counts(p,y);hp+=h1;hn+=h0
   for k,v in c.items():summed[k]=summed.get(k,0)+v
   row=dict(name=name,category=cat,counts=c,metrics=metrics(c,h1,h0),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),label_sha256=hashlib.sha256(y.tobytes()).hexdigest())
   if key.startswith('v2'):row['geometry']=metadata(name)
   rows.append(row)
   # Fixed slice, fixed sample choice, independent of performance.
   visual=(key=='v2_val' and name in {next(n for n,k in ds.items if k==ct) for ct in CATEGORIES}) or (key in ('faultseg3d20','wu_style_test') and i<2) or (style!='clean' and name==sorted(subset)[0])
   if visual:
    np.savez_compressed(dest/f'{key}__{name}.npz',seismic=x[:,64,:],label=y[:,64,:],probability=p[:,64,:])
   if len(rows)%10==0:print(a.model,key,len(rows),'seconds',round(time.time()-start),flush=True)
   if a.smoke:break
  save_json(dest/(key+'.json'),dict(dataset=key,style=style,n=len(rows),seconds=time.time()-start,counts=summed,metrics=metrics(summed,hp,hn),hist_pos=hp.tolist(),hist_neg=hn.tolist(),volumes=rows))
  print('DONE',a.model,key,metrics(summed,hp,hn),flush=True)
 save_json(dest/'complete.json',dict(finished=time.strftime('%Y-%m-%d %H:%M:%S'),tasks=len(tasks)))
if __name__=='__main__':main()
