"""Render selected three-column rows independently and assemble the gallery."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/faultsyn_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,time,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qc.render3d import figure_gallery3
from PIL import Image


def render(job):
    root,out,name,wait=job;root=Path(root);out=Path(out)
    deadline=time.monotonic()+wait
    while not (root/'metadata'/f'{name}.json').exists():
        if time.monotonic()>deadline:raise RuntimeError(f'Missing completed sample {name}')
        time.sleep(5)
    path=out/'rows'/f'{name}.png'
    if not path.exists():figure_gallery3(root,[name],(128,128,128),path)
    return name


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--wait-seconds',type=int,default=0);a=p.parse_args()
    names=(a.out/'picked.txt').read_text().split();(a.out/'rows').mkdir(parents=True,exist_ok=True)
    jobs=[(str(a.root),str(a.out),n,a.wait_seconds) for n in names]
    jobs.sort(key=lambda j:not(a.root/'metadata'/f'{j[2]}.json').exists())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(render,j) for j in jobs]):print(f.result(),flush=True)
    imgs=[Image.open(a.out/'rows'/f'{n}.png').convert('RGB') for n in names]
    w=max(im.width for im in imgs);result=Image.new('RGB',(w,sum(im.height for im in imgs)),'white');y=0
    for im in imgs:result.paste(im,((w-im.width)//2,y));y+=im.height
    result.save(a.out/'gallery_3col.png');print(a.out/'gallery_3col.png',flush=True)
if __name__=='__main__':main()
