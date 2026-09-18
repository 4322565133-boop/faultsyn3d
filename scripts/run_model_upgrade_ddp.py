"""Sequential four-GPU jobs: upgraded model first, matched control second."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/model_upgrade_ddp_20260916')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/'processes.json'
    if manifest.exists() and not a.resume:
        raise RuntimeError('DDP pilot already exists; inspect or explicitly resume individual run')
    plan = {'supervisor_pid': os.getpid(), 'epochs': a.epochs, 'started_unix': time.time(),
            'order': ['combined', 'baseline'], 'world_size': 4, 'global_batch': 8, 'runs': []}
    if a.resume:
        plan['previous_attempts'] = json.loads(manifest.read_text())
        plan['resumed_unix'] = time.time()
    manifest.write_text(json.dumps(plan, indent=2))
    for variant in plan['order']:
        if a.resume and (root/variant/'test.json').exists():
            continue
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=4',
               '--master_addr=127.0.0.1', '--master_port=29617', 'train/train_model_upgrade_ddp.py',
               '--variant', variant, '--out', str(root/variant), '--epochs', str(a.epochs)]
        if a.resume and (root/variant/'last.pt').exists():
            cmd += ['--resume', '--resume-numerics-fix']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with open(root/f'{variant}.log', 'a' if a.resume else 'w') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            record = {'variant': variant, 'pid': child.pid, 'command': cmd, 'started_unix': time.time()}
            plan['runs'].append(record)
            manifest.write_text(json.dumps(plan, indent=2))
            print(f'START {variant} launcher_pid={child.pid}', flush=True)
            record['returncode'] = child.wait()
            record['finished_unix'] = time.time()
            manifest.write_text(json.dumps(plan, indent=2))
            if record['returncode']:
                status_file = root/variant/'status.json'
                state = json.loads(status_file.read_text()) if status_file.exists() else {}
                state.update(status='failed', returncode=record['returncode'],
                             failed_unix=time.time(), log=str(root/f'{variant}.log'))
                status_file.write_text(json.dumps(state, indent=2))
                (root/'pipeline_status.json').write_text(json.dumps({'status': 'failed',
                    'variant': variant, 'returncode': record['returncode']}, indent=2))
                print(f'FAILED {variant}: inspect {root/variant}.log', flush=True)
                raise SystemExit(record['returncode'])
    subprocess.run([sys.executable, 'scripts/report_model_upgrade.py', '--root', str(root)],
                   env=env, check=True)
    (root/'pipeline_status.json').write_text(json.dumps({'status': 'complete'}, indent=2))
    print(f'COMPLETE: {root}/comparison.md', flush=True)


if __name__ == '__main__':
    main()
