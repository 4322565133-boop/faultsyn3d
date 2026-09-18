"""Launch four isolated GPU runs and write their final controlled comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(root):
    reference = .8325868632244817
    rows = []
    for variant in ['baseline', 'architecture', 'loss', 'combined']:
        path = root/variant/'test.json'
        if path.exists():
            row = json.loads(path.read_text())
            rows.append(row)
    baseline = next((r['iou'] for r in rows if r['variant'] == 'baseline'), None)
    for row in rows:
        row['delta_vs_matched_baseline'] = row['iou']-baseline if baseline is not None else None
    report = {'kind': '20-epoch warm-start pilot; single seed; not a from-scratch comparison',
              'historical_test_iou': reference, 'rows': rows, 'complete': len(rows) == 4}
    (root/'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# MaxViT upgrade pilot', '', 'All groups start from the same checkpoint; selected by validation IoU.', '',
             '| Variant | Best epoch | Test IoU | Dice | Delta vs historical (pp) | Delta vs matched control (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        delta = r['delta_vs_matched_baseline']
        text = f'{100*delta:+.3f}' if delta is not None else 'pending'
        lines.append(f'| {r["variant"]} | {r["epoch"]} | {r["iou"]:.6f} | {r["dice"]:.6f} | '
                     f'{100*(r["iou"]-reference):+.3f} | {text} |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/model_upgrade_pilot_20260916')
    p.add_argument('--epochs', type=int, default=20)
    a = p.parse_args()
    os.chdir(ROOT)
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'processes.json').exists():
        raise RuntimeError('Pilot already launched; inspect existing processes instead of duplicating')
    children, handles, records = [], [], []
    for gpu, variant in enumerate(['baseline', 'architecture', 'loss', 'combined']):
        log = open(root/f'{variant}.log', 'w')
        cmd = [sys.executable, '-u', 'train/train_model_upgrade.py', '--variant', variant,
               '--gpu', str(gpu), '--out', str(root/variant), '--epochs', str(a.epochs)]
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        children.append(child); handles.append(log)
        records.append({'variant': variant, 'gpu': gpu, 'pid': child.pid, 'command': cmd})
    (root/'processes.json').write_text(json.dumps({'supervisor_pid': os.getpid(), 'epochs': a.epochs,
                                                 'started_unix': time.time(), 'runs': records}, indent=2))
    print(json.dumps(records), flush=True)
    while any(c.poll() is None for c in children):
        time.sleep(30)
    status = dict(zip([r['variant'] for r in records], [c.returncode for c in children]))
    for handle in handles:
        handle.close()
    (root/'exit_codes.json').write_text(json.dumps(status, indent=2))
    report = summarize(root)
    print(json.dumps({'exit_codes': status, 'comparison': report}, indent=2), flush=True)
    if any(status.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
