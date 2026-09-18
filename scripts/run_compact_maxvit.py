"""Detached supervisor: train, update live comparison, then evaluate once.

SIGTERM stops the training process group and prevents subsequent evaluation.
Training failures remain visible; they do not silently advance to another job.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qc.compact_live import render


def write_json(obj,path):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='runs/compact_maxvit_v2_m0_20260916')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();os.chdir(ROOT);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'processes.json').exists() and not a.resume:
        raise RuntimeError('Existing launch; inspect it before using --resume')
    reference=ROOT/'runs/maxvit3d_v2_oldrecipe_all3'
    if not (out/'reference_maxvit_log.csv').exists():
        shutil.copy2(reference/'log.csv',out/'reference_maxvit_log.csv')
        shutil.copy2(reference/'args.json',out/'reference_maxvit_args.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/faultsyn_compact_mpl')
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4','--master_addr=127.0.0.1',
         '--master_port=29673','train/train_compact_maxvit_ddp.py','--out',str(out)]
    if a.resume:cmd.append('--resume')
    child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    record={'supervisor_pid':os.getpid(),'started':time.time(),'command':cmd,'world_size':4,'out':str(out)}
    with (out/'train.log').open('a' if a.resume else 'w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record['training_pid']=child.pid;write_json(record,out/'processes.json')
        while child.poll() is None:
            try:render(out)
            except Exception as e:print('Live report error:',repr(e),flush=True)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        record['train_returncode']=child.returncode;record['training_finished']=time.time()
        write_json(record,out/'processes.json')
    if stopping or child.returncode:
        write_json({'state':'stopped' if stopping else 'failed','returncode':child.returncode,'updated':time.time()},out/'status.json')
        render(out);return
    # No repeated test-set monitoring while training.
    with (out/'test_evaluation.log').open('w') as log:
        cmd=[sys.executable,'train/evaluate.py','--gpu','0','--runs',str(out)]
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record['evaluation_pid']=child.pid;write_json(record,out/'processes.json')
        record['eval_returncode']=child.wait();write_json(record,out/'processes.json')
    write_json({'state':'complete' if not stopping and child.returncode==0 else 'evaluation_failed',
                'updated':time.time(),'train_epochs':200},out/'status.json')
    render(out)


if __name__=='__main__':main()
