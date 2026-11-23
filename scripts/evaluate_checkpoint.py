"""Portable re-evaluation of a chosen checkpoint, always to a new local output folder."""
import argparse,hashlib,json,csv,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
import torch
from PIL import Image
from models import UNet
from dataset import PairDataset
from common import PALETTE,save_json
from final_test import image_statistics,aggregate
p=argparse.ArgumentParser();p.add_argument('--model',choices=['baseline','attention'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='cpu',choices=['cpu','mps']);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int);args=p.parse_args()
assert not args.out.exists(),'Choose a new output folder to preserve archived results'
weight=args.checkpoint or R/f'runs/formal_{args.model}_seed42/best.pt'
hash_=hashlib.sha256(weight.read_bytes()).hexdigest();saved=torch.load(weight,map_location='cpu',weights_only=False)
protocol=json.loads((R/'configs/final-test-approved.json').read_text());names=protocol['test_filenames']
if args.limit is not None:assert 0<args.limit<=1929;names=names[:args.limit]
ds=PairDataset('test',names,augment=False)
for key in ['mean','std']:setattr(ds,key,np.array(saved['config']['normalization'][key],dtype=np.float32))
torch.set_num_threads(2);model=UNet(attention=args.model=='attention').to(args.device);model.load_state_dict(saved['model']);model.eval()
args.out.mkdir(parents=True);rgb=args.out/'predictions_rgb';ids=args.out/'predictions_class_id';rgb.mkdir();ids.mkdir();rows=[];matrices=[]
with torch.no_grad():
 for i in range(0,len(ds),2):
  items=[ds[j] for j in range(i,min(i+2,len(ds)))];x=torch.stack([v[0] for v in items]).to(args.device);pred=model(x).argmax(1).cpu().numpy().astype(np.uint8)
  for j,(v,y) in enumerate(items):
   name=names[i+j];a,cm=image_statistics(name,y.numpy().astype(np.uint8),pred[j]);rows.append(a);matrices.append(cm)
   Image.fromarray(pred[j]).save(ids/name);Image.fromarray(PALETTE[pred[j]]).save(rgb/name)
summary,ranked=aggregate(rows,matrices);summary.update(model=args.model,checkpoint_sha256=hash_,device=args.device,partial_diagnostic=len(rows)!=1929)
for filename,data in [('per-image.csv',rows),('ranked-images.csv',ranked)]:
 with (args.out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
save_json(args.out/'summary.json',summary)
print(json.dumps({'samples':len(rows),'foreground_miou':summary['foreground_miou'],'partial_diagnostic':summary['partial_diagnostic']}))
