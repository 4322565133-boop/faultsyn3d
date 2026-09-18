"""Analytic thick prediction regression for actual legacy evaluator and audit metrics."""
import json
import numpy as np
import torch
from train.eval_cross import run
from scripts.audit_cross_domain import counts,metrics,ROOT
p=np.zeros((9,9,9),np.float32);p[3:6,3:6,3:6]=1
y=np.zeros_like(p);y[4,4,4]=1;y[0,0,0]=1
c,hp,hn=counts(p,y);a=metrics(c,hp,hn)
x=torch.from_numpy(np.where(p>0,20.,-20.).astype(np.float32))[None,None]
target=torch.from_numpy(y)[None,None]
b=run(torch.nn.Identity(),[(x,target,0)],torch.device('cuda:3'))
for k in ['iou','precision','recall','ap','tol2_precision','tol2_recall']:assert abs(a[k]-b[k])<1e-12,(k,a[k],b[k])
assert b['tol2_recall']==.5 and b['tol2_precision']==1
(ROOT/'metric_regression.json').write_text(json.dumps({'status':'PASS','case':'27 predicted voxels near first of 2 GT points; tolerance recall must be 1/2, not 27/28','metrics':b},indent=2))
print('PASS')
