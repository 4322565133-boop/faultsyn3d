"""Four GPUs per job: new grid model, then fresh matched U-Net-L control."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qc.unet_grid_live import render


def save(x,p):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/unet_grid_pair_20260916')
    p.add_argument('--resume',action='store_true');a=p.parse_args();os.chdir(ROOT)
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'processes.json').exists() and not a.resume:raise RuntimeError('Existing experiment; inspect before --resume')
    for src,dst in [('runs/maxvit3d_unet_l_ddp/log.csv','reference_unet_l.csv'),('runs/maxvit3d_v2_oldrecipe_all3/log.csv','reference_maxvit.csv')]:
        if not (root/dst).exists():shutil.copy2(src,root/dst)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/faultsyn_unet_grid_mpl')
    record={'supervisor_pid':os.getpid(),'started':time.time(),'order':['grid','baseline'],'jobs':[]}
    if a.resume:record['previous']=json.loads((root/'processes.json').read_text())
    child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for variant in record['order']:
        out=root/variant;out.mkdir(exist_ok=True)
        if a.resume and (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('state')=='complete':continue
        if stopping:break
        cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4','--master_addr=127.0.0.1',
             '--master_port=29675','train/train_unet_grid_ddp.py','--variant',variant,'--out',str(out)]
        if a.resume and (out/'last.pt').exists():cmd.append('--resume')
        with (out/'train.log').open('a' if a.resume else 'w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            job={'variant':variant,'pid':child.pid,'command':cmd};record['jobs'].append(job);save(record,root/'processes.json')
            save({'state':'training '+variant,'active':variant,'updated':time.time()},root/'pipeline_status.json')
            while child.poll() is None:
                try:render(root)
                except Exception as e:print('Report error',repr(e),flush=True)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            job['returncode']=child.returncode;save(record,root/'processes.json')
        if stopping or child.returncode:
            save({'state':'stopped' if stopping else 'failed','active':variant,'returncode':child.returncode,'updated':time.time()},root/'pipeline_status.json')
            render(root);return
    if stopping:return
    # One test-set evaluation only after both models finish selection on validation.
    save({'state':'final test evaluation','updated':time.time()},root/'pipeline_status.json')
    cmd=[sys.executable,'train/evaluate.py','--gpu','0','--runs',str(root/'grid'),str(root/'baseline')]
    with (root/'test_evaluation.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record['evaluation_pid']=child.pid;save(record,root/'processes.json');code=child.wait()
    if code or stopping:
        save({'state':'evaluation failed or stopped','returncode':code,'updated':time.time()},root/'pipeline_status.json');render(root);return
    result={variant:json.loads((root/variant/'test.json').read_text()) for variant in record['order']}
    result['delta_grid_minus_baseline']={key:result['grid'][key]-result['baseline'][key] for key in ['iou','dice','precision','recall','ap']}
    save(result,root/'comparison.json')
    text='# U-Net-L 与 Grid Attention 最终对照\n\n|模型|IoU|Dice|P|R|AP|\n|---|---:|---:|---:|---:|---:|\n'
    for v in record['order']:text+='|'+v+'|'+ '|'.join(f'{result[v][k]:.6f}' for k in ['iou','dice','precision','recall','ap'])+'|\n'
    text+='\n单随机种子配对实验；仍需消融和重复实验。测试期间无辅助几何输入。\n'
    (root/'comparison.md').write_text(text)
    save({'state':'complete','updated':time.time()},root/'pipeline_status.json');render(root)


if __name__=='__main__':main()
