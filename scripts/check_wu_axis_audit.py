import json
import numpy as np
import torch
from scripts.audit_cross_domain import ROOT,RUNS,counts,metrics
from train.analyze_loss_vs_iou import load
from train.eval_cross import FaultSeg3DVal

torch.set_num_threads(4);dev=torch.device('cuda:3');ds=FaultSeg3DVal();out={}
for name,run in RUNS.items():
 m,ep=load(run,dev);total={};hp=np.zeros(2000);hn=hp.copy()
 for x,y,_ in ds:
  # Deliberately restore raw-file order: depth last. Diagnostic only.
  x=x.permute(0,3,2,1).contiguous();y=y[0].numpy().transpose(2,1,0)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=m(x[None].to(dev)).float().sigmoid()[0,0].cpu().numpy()
  c,a,b=counts(p,y);hp+=a;hn+=b
  for k,v in c.items():total[k]=total.get(k,0)+v
 out[name]=dict(epoch=ep,axis='raw-file depth-last, diagnostic not selected for score',metrics=metrics(total,hp,hn));print(name,out[name],flush=True)
 del m;torch.cuda.empty_cache()
(ROOT/'axis_diagnostic.json').write_text(json.dumps(out,indent=2))
