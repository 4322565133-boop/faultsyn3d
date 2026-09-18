"""Read-only experiment inventory and figures for the September 18 model proposal.

Training histories are observations, not independent trials. Historical test.json
files are retained as unverified artifacts because some were overwritten by OOD
evaluation. No model ranking is inferred across incompatible protocols.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final_model_design_20260918'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, hashes = [], {}
    for path in sorted(list((ROOT/'runs').rglob('log.csv')) + list((ROOT/'runs').rglob('history.json'))):
        cfg = read(path.parent/'config.json')
        args = read(path.parent/'args.json') or cfg.get('args', cfg)
        if path.suffix == '.csv':
            rows = list(csv.DictReader(path.open()))
            scored = [(r, {k: number(r.get('val_'+k)) for k in ('iou', 'precision', 'recall')}) for r in rows]
            metric_source = 'val_iou from CSV; checkpoint selection may instead use val_loss'
        else:
            rows = read(path)
            scored = [(r, r.get('full_val') or {}) for r in rows]
            metric_source = 'full_val; fast_val excluded'
        scored = [(r, m) for r, m in scored if number(m.get('iou')) is not None]
        best, metrics = max(scored, key=lambda x: float(x[1]['iou'])) if scored else ({}, {})
        data = args.get('root', args.get('data', cfg.get('data', 'unknown')))
        if args.get('legacy'):
            data = 'legacy dataset (overrides args.data; see training loader)'
        elif args.get('wu'):
            data = 'local Wu-style synthetic dataset (overrides args.data)'
        sources = [path, path.parent/'config.json', path.parent/'args.json', path.parent/'test.json']
        for source in sources:
            if source.exists():
                hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        record = dict(
            run=str(path.parent.relative_to(ROOT)), history=str(path.relative_to(ROOT)),
            data=data, history_rows=len(rows),
            last_epoch=max((number(r.get('epoch')) or 0 for r in rows), default=0),
            configured_epochs=args.get('epochs'),
            best_recorded_val_iou=metrics.get('iou'), best_val_epoch=best.get('epoch'),
            best_val_precision=metrics.get('precision'), best_val_recall=metrics.get('recall'),
            model=cfg.get('model', args.get('model', args.get('variant', 'unspecified'))),
            parameters_recorded=cfg.get('params', cfg.get('parameters', args.get('params'))),
            protocol_hash=cfg.get('protocol_sha256', args.get('data_source_sha256', cfg.get('manifest_sha256'))),
            metric_source=metric_source,
            test_json_status='unverified provenance; do not treat as in-domain automatically' if (path.parent/'test.json').exists() else 'absent',
        )
        records.append(record)
    (OUT/'experiment_inventory.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), count=len(records),
        warning='Rows are log files, not independent completed experiments. Best logged IoU is not necessarily the selected checkpoint.',
        records=records), indent=2))
    with (OUT/'experiment_inventory.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    results = []
    for suffix, name in [('unet_l','UNet-L'), ('maxvit_tiny','MaxViT'), ('mv2d_tiny_ad','2D MaxViT + adapter')]:
        p = OUT/'diagnostics'/f'thebe_spatial_v3_{suffix}_best_fullval_full.json'
        d=read(p)
        if not d:
            continue
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        d['display_name']=name
        strata=d['strata']
        d['fp_fraction_in_positive_label_cores']=strata['positive_label_core']['fp'] / (strata['positive_label_core']['fp']+strata['empty_label_core']['fp'])
        results.append(d)
    if len(results)==3:
        m=results[1]['strata']['positive_label_core']; a=results[2]['strata']['positive_label_core']
        extra_tp=a['tp']-m['tp']
        extra_fp=sum(results[2]['strata'][k]['fp']-results[1]['strata'][k]['fp'] for k in ('positive_label_core','empty_label_core'))
        (OUT/'diagnostic_summary.json').write_text(json.dumps(dict(
            note='All metrics on the same validation support, AMP inference; AP approximated with 2001 histogram bins. Validation-selected threshold scores are optimistic diagnostics, not new held-out scores.',
            results=results, adapter_vs_maxvit=dict(extra_tp=extra_tp, extra_fp=extra_fp, extra_fp_per_extra_tp=extra_fp/extra_tp)),indent=2))
        plot(results)
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(history_files=len(records), csv_logs=sum(r['history'].endswith('.csv') for r in records), json_histories=sum(r['history'].endswith('.json') for r in records), diagnostic_runs=len(results)),indent=2))


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#3975ac','#e38327','#3b9473']
    fig, axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for d,c in zip(results,colors):
        g=d['threshold_grid']; name=d['display_name']
        axes[0,0].plot([x['threshold'] for x in g],[x['iou'] for x in g],color=c,label=name)
        axes[0,1].plot([x['recall'] for x in g],[x['precision'] for x in g],color=c,label=f"{name}: AP~{d['histogram_ap']:.3f}")
        a=d['at_0_5']; axes[0,1].scatter([a['recall']],[a['precision']],color=c)
    axes[0,0].set(title='Validation IoU versus threshold',xlabel='Threshold',ylabel='IoU')
    axes[0,0].axvline(.5,color='grey',ls='--',lw=1);axes[0,0].legend()
    axes[0,1].set(title='Threshold-grid PR (dots: threshold 0.5)',xlabel='Recall',ylabel='Precision')
    axes[0,1].legend()
    x=np.arange(3); names=[d['display_name'] for d in results]
    tp=[d['strata']['positive_label_core']['tp']/1e6 for d in results]
    fp=[(d['strata']['positive_label_core']['fp']+d['strata']['empty_label_core']['fp'])/1e6 for d in results]
    fn=[d['strata']['positive_label_core']['fn']/1e6 for d in results]
    for off,vals,label,c in [(-.25,tp,'True positive','#3b9473'),(0,fp,'False positive','#b64b52'),(.25,fn,'False negative','#8e88b5')]:
        axes[1,0].bar(x+off,vals,.23,label=label,color=c)
    axes[1,0].set(xticks=x,xticklabels=names,ylabel='Million voxels',title='Full validation counts at threshold 0.5');axes[1,0].legend()
    for off,key,label,c in [(-.16,'padded_context','Padded context','#91adc5'),(.16,'interior_context','Interior context','#446b89')]:
        axes[1,1].bar(x+off,[d['strata'][key]['iou'] for d in results],.30,label=label,color=c)
    axes[1,1].set(xticks=x,xticklabels=names,ylabel='IoU',title='Spatial strata (different geology; not a causal test)');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Thebe spatial-v3 | 2,433 validation cores | No test evaluation\nUNet / MaxViT: best epoch 47 of 100; adapter: best epoch 16 of 24',fontsize=13)
    fig.savefig(OUT/'validation_diagnosis.png',dpi=150)
    fig.savefig(OUT/'validation_diagnosis.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
