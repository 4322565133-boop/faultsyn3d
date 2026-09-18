"""Detached single-run supervisor. No test evaluation or automatic experiment queue."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/thebe_spatial_v3_unet_l_trial'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--supervise',action='store_true')
    p.add_argument('--resume',action='store_true'); p.add_argument('--stop-after',type=int,default=30); a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if not a.supervise:
        marker=RUN/'supervisor.json'
        if marker.exists():
            old=json.loads(marker.read_text())
            if old.get('status')=='running':
                try: os.kill(old['supervisor_pid'],0)
                except ProcessLookupError: pass
                else: raise RuntimeError('Supervisor is still running; refusing duplicate launch')
        if (RUN/'last.pt').exists() and not a.resume: raise RuntimeError('Existing checkpoint requires --resume')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--supervise','--stop-after',str(a.stop_after)]
        if a.resume: cmd.append('--resume')
        with open(RUN/'supervisor.log','a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'supervisor_pid':child.pid,'run':str(RUN)})); return
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0,1,2,3'; env['OMP_NUM_THREADS']='2'
    cmd=[sys.executable,'-m','torch.distributed.run','--master_addr','127.0.0.1','--master_port','29862',
        '--nproc_per_node','4','--','train/train_thebe_spatial_ddp.py','--run',str(RUN),
        '--epochs','100','--stop-after',str(a.stop_after),'--samples','2400','--workers','2']
    if a.resume: cmd.append('--resume')
    state=dict(status='running',supervisor_pid=os.getpid(),command=cmd,started=datetime.datetime.now().astimezone().isoformat())
    with open(RUN/'train.log','a') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['torchrun_pid']=process.pid; (RUN/'supervisor.json').write_text(json.dumps(state,indent=2))
        code=process.wait()
    state.update(status='completed' if code==0 else 'failed',returncode=code,ended=datetime.datetime.now().astimezone().isoformat())
    (RUN/'supervisor.json').write_text(json.dumps(state,indent=2))
    sys.exit(code)

if __name__=='__main__': main()
