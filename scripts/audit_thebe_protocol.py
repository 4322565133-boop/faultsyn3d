"""Header/manifest audit only: no target-guided sample or model selection."""
import json,zipfile,hashlib
from pathlib import Path
import numpy as np
p=Path('data/thebe');out=Path('reports/thebe_model_design_20260917');headers={}
for kind in ['seis','fault']:
 headers[kind]={}
 for f in sorted((p/kind).glob('*.npz')):
  with zipfile.ZipFile(f) as z:
   with z.open(z.namelist()[0]) as a:
    shape,order,dtype=np.lib.format._read_array_header(a,np.lib.format.read_magic(a))
  headers[kind][f.stem.removeprefix(kind)]={'shape':list(shape),'dtype':str(dtype),'compressed_bytes':f.stat().st_size}
assert headers['seis'].keys()==headers['fault'].keys()
for k in headers['seis']:assert headers['seis'][k]['shape']==headers['fault'][k]['shape']
mfile=Path('data/thebe_cubes/manifest.json');m=json.loads(mfile.read_text());splits={};offsets={'train':0,'validation':900,'test':1100};lengths={'train':900,'validation':200,'test':703}
for split,nxl in lengths.items():
 rows=[r for r in m['volumes'] if r['split']==split];orig=np.array([r['origin'] for r in rows]);fr=np.array([r['fault_fraction'] for r in rows])
 assert np.all(orig>=0) and np.all(orig+128<=np.array([1537,3174,nxl]))
 assert set(m[split])=={r['name'] for r in rows}
 splits[split]={'n_cubes':len(rows),'unique_origins':len(set(map(tuple,orig))),'origin_min':orig.min(0).tolist(),'origin_max':orig.max(0).tolist(),'global_section_extent_half_open':[int(orig[:,2].min()+offsets[split]),int(orig[:,2].max()+offsets[split]+128)],'mean_positive_fraction':float(fr.mean()),'density_bins_counts':[int((fr<.005).sum()),int(((fr>=.005)&(fr<.02)).sum()),int((fr>=.02).sum())]}
result={'headers':headers,'spatial_split_checks':'PASS: all stored cubes wholly inside original disjoint train/val/test section ranges','manifest_sha256':hashlib.sha256(mfile.read_bytes()).hexdigest(),'splits':splits,'limitations':['Header audit does not check every voxel or ZIP CRC.','900/200/703 adjacent spatial sections belong to one survey; not independent cross-survey generalization.','Split has no deliberate guard gap; report adjacent-region protocol honestly.','600 validation patches overlap and are not 600 independent geological volumes.'],'protocol_issues':['depth_band uses each split ground truth positives to choose evaluation ROI','validation strata documented as matching test distribution','full-volume tolerance recall implementation uses predicted-match numerator','fault_none means <0.5% label fraction, not necessarily truly empty','source header float64 differs from module docstring float32; explicit conversion exists']}
(out/'data_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(splits,indent=2))
