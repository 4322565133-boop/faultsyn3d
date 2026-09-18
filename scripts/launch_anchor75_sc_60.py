"""User-authorized sequential UNet -> SC-MaxViT queue, each capped at 60 epochs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'runs/thebe_anchor75q_60_queue'
RUNS=[('unet_l',ROOT/'runs/thebe_anchor75q_60_unet_l'),('sc_maxvit',ROOT/'runs/thebe_anchor75q_60_sc_maxvit')]


def stamp():return datetime.datetime.now().astimezone().isoformat()


def write(state):
    tmp=QUEUE/'queue.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(QUEUE/'queue.json')


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for (name,path),color in zip(RUNS,('#3975ac','#d95835')):
        try:hist=json.loads((path/'history.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):continue
        for ax,key in zip(axes.flat,('iou','ap','precision','recall')):
            rows=[r for r in hist if r.get('full_val')]
            if rows:ax.plot([r['epoch'] for r in rows],[r['full_val'][key] for r in rows],label=name,color=color)
    axes[0,0].axhline(.3953668213,color='grey',ls='--',lw=1,label='Historical MaxViT best (uniform, 100-epoch schedule)')
    for ax,key in zip(axes.flat,('IoU @ 0.5','AP (2001 bins)','Precision @ 0.5','Recall @ 0.5')):
        ax.set(xlabel='Epoch',ylabel=key,xlim=(0,60),ylim=(0,1));ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    fig.suptitle('Thebe | shared anchor75 plans | full validation only | TEST not used\nHistorical line is a reference under a different training recipe')
    dest=ROOT/'runs/anchor75q_60_live.png';tmp=dest.with_suffix('.tmp.png');fig.savefig(tmp,dpi=130);plt.close(fig);tmp.replace(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--supervise',action='store_true')
    p.add_argument('--per-rank',type=int,default=2);p.add_argument('--accum',type=int,default=1)
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    QUEUE.mkdir(parents=True,exist_ok=True)
    if not a.supervise:
        if (QUEUE/'queue.json').exists():
            s=json.loads((QUEUE/'queue.json').read_text())
            if s.get('status')=='running':
                try:os.kill(s['supervisor_pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('Queue supervisor already running')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--supervise','--per-rank',str(a.per_rank),'--accum',str(a.accum),'--workers',str(a.workers)]
        env=os.environ.copy();env['MPLCONFIGDIR']=str(QUEUE/'mpl_cache')
        with (QUEUE/'supervisor.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,queue=str(QUEUE),models=[n for n,_ in RUNS],epochs_each=60)));return
    lock=(QUEUE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',supervisor_pid=os.getpid(),started=stamp(),order=[n for n,_ in RUNS],epochs_each=60,events=[])
    write(state)
    try:
        for index,(name,path) in enumerate(RUNS):
            path.mkdir(parents=True,exist_ok=True)
            if (path/'status.json').exists():
                previous=json.loads((path/'status.json').read_text())
                if previous.get('status')=='completed' and previous.get('last_completed_epoch')==60:
                    state['events'].append(dict(model=name,status='already_completed',time=stamp()));continue
            for attempt in range(3):
                cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node','4','--master_addr','127.0.0.1',
                    '--master_port',str(29931+index),'--','train/train_thebe_anchor_ddp.py','--run',str(path),
                    '--model',name,'--epochs','60','--stop-after','60','--workers',str(a.workers),
                    '--per-rank',str(a.per_rank),'--accum',str(a.accum)]
                if (path/'last.pt').exists():cmd.append('--resume')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                    TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONUNBUFFERED='1')
                with (path/'train.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state.update(current_model=name,attempt=attempt+1,torchrun_pid=child.pid,command=cmd);write(state)
                    while True:
                        try:code=child.wait(timeout=30);break
                        except subprocess.TimeoutExpired:
                            state['heartbeat']=stamp();write(state)
                            try:draw()
                            except Exception as ex:print('plot warning',repr(ex),flush=True)
                state['events'].append(dict(model=name,attempt=attempt+1,returncode=code,time=stamp()));write(state);draw()
                if code==0:
                    done=json.loads((path/'status.json').read_text())
                    if done.get('status')!='completed' or done.get('last_completed_epoch')!=60:
                        raise RuntimeError(f'{name} exited without completing 60 epochs')
                    break
                if attempt==2:raise RuntimeError(f'{name} failed three attempts; see {path}/train.log')
                time.sleep(10)
        state.update(status='completed',ended=stamp());write(state);draw()
    except BaseException as ex:
        state.update(status='failed',error=repr(ex),ended=stamp());write(state)
        raise


if __name__=='__main__':main()
