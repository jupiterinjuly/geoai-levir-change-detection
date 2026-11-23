"""Post-segment validation analysis. Does not access the test image split.

These outputs support diagnostics/application preparation and must never be
presented as final test performance. Final test evaluation is a later stage.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from common import ROOT,EVIDENCE,PALETTE,confusion,metrics,save_json
from formal_data import FormalDataset
from formal_train import sync
from models import UNet

def image_statistics(filename,target,prediction):
    matrix=confusion(target,prediction);n=target.size
    actual=np.bincount(target.reshape(-1),minlength=3)
    predicted=np.bincount(prediction.reshape(-1),minlength=3)
    row={'filename':filename,'pixels':n,'true_change_pixels':int(actual[1:].sum()),
        'pred_change_pixels':int(predicted[1:].sum())}
    for c,label in [(1,'road'),(2,'building')]:
        tp=int(matrix[c,c]);fp=int(matrix[:,c].sum()-tp);fn=int(matrix[c,:].sum()-tp)
        row.update({f'true_{label}_pixels':int(actual[c]),f'pred_{label}_pixels':int(predicted[c]),
            f'true_{label}_fraction':float(actual[c]/n),f'pred_{label}_fraction':float(predicted[c]/n),
            f'{label}_area_abs_error_pp':float(100*abs(actual[c]/n-predicted[c]/n)),
            f'{label}_tp':tp,f'{label}_fp':fp,f'{label}_fn':fn})
    fraction=row['true_change_pixels']/n
    row['size_group']='empty' if fraction==0 else 'small_0_to_1pct' if fraction<=.01 else 'medium_1_to_10pct' if fraction<=.1 else 'large_over_10pct'
    return row,matrix

def aggregate(rows,matrices):
    matrix=np.sum(matrices,axis=0)
    positive=[r for r in rows if r['true_change_pixels']>0]
    count=math.ceil(.2*len(rows))
    ranked=sorted(rows,key=lambda r:(-r['pred_change_pixels']/r['pixels'],r['filename']))
    selected=ranked[:count]
    total_pixels=sum(r['true_change_pixels'] for r in rows)
    empty=[r for r in rows if r['true_change_pixels']==0]
    summary={'samples':len(rows),**metrics(matrix),
        'road_area_mae_pp':float(np.mean([r['road_area_abs_error_pp'] for r in rows])),
        'building_area_mae_pp':float(np.mean([r['building_area_abs_error_pp'] for r in rows])),
        'empty_images':len(empty),
        'empty_false_alarm_image_rate':sum(r['pred_change_pixels']>0 for r in empty)/len(empty) if empty else None,
        'empty_mean_pred_change_fraction':float(np.mean([r['pred_change_pixels']/r['pixels'] for r in empty])) if empty else None,
        'ranking_top20pct':{'selected_images':count,'positive_images_total':len(positive),
            'positive_image_recall':sum(r['true_change_pixels']>0 for r in selected)/len(positive) if positive else None,
            'true_change_pixel_coverage':sum(r['true_change_pixels'] for r in selected)/total_pixels if total_pixels else None,
            'tie_break':'filename ascending','claim_scope':'offline scale-based sorting; not measured human labor savings'}}
    groups={}
    for group in ['empty','small_0_to_1pct','medium_1_to_10pct','large_over_10pct']:
        indices=[i for i,r in enumerate(rows) if r['size_group']==group]
        groups[group]={'images':len(indices),'metrics':metrics(np.sum([matrices[i] for i in indices],axis=0)) if indices else None}
    summary['size_groups']=groups
    summary['size_group_definition']='fixed before validation analysis: zero, (0,1%], (1%,10%], >10% true foreground pixels'
    return summary,ranked

def main(args):
    torch.set_num_threads(2)
    folder=ROOT/'runs'/f'formal_{args.model}_seed42'
    run=json.loads((folder/'summary.json').read_text())
    if (run['epochs_completed'] != args.segment_cap and not run['early_stopped']) or run['smoke_only']:
        raise ValueError('Run this only after the approved segment finishes or early stops')
    checkpoint_path=folder/'best.pt';checkpoint_hash=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=checkpoint['config'];names=config['validation_filenames']
    dataset=FormalDataset('val',names,config['normalization'])
    device=torch.device('mps');model=UNet(attention=args.model=='attention').to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    out=folder/f'validation_analysis_best_ep{checkpoint["epoch"]:02d}_{checkpoint_hash[:12]}'
    out.mkdir(parents=True,exist_ok=True)
    rgb_folder=out/'predictions_rgb';id_folder=out/'predictions_class_id'
    rgb_folder.mkdir(exist_ok=True);id_folder.mkdir(exist_ok=True)
    rows=[];matrices=[];started=time.perf_counter()
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(1000042))
    with torch.no_grad():
        for index,(x,y) in enumerate(loader):
            logits=model(x.to(device));predictions=logits.argmax(1).cpu().numpy().astype(np.uint8)
            targets=y.numpy().astype(np.uint8)
            for j,(target,prediction) in enumerate(zip(targets,predictions)):
                filename=names[index*2+j]
                row,matrix=image_statistics(filename,target,prediction);rows.append(row);matrices.append(matrix)
                Image.fromarray(PALETTE[prediction]).save(rgb_folder/filename)
                Image.fromarray(prediction).save(id_folder/filename)
            if (index+1)%100==0:print('validation analyzed',len(rows),'of',len(dataset),flush=True)
    sync(device)
    summary,ranked=aggregate(rows,matrices)
    reference=checkpoint['validation']
    if abs(summary['foreground_miou']-reference['foreground_miou'])>1e-5:
        raise ValueError('Reloaded checkpoint validation differs from selection record')
    summary.update({'model':args.model,'split':'val','checkpoint_epoch':checkpoint['epoch'],
        'checkpoint_sha256':checkpoint_hash,'evaluation_seconds_including_io_and_png':time.perf_counter()-started,
        'selected_validation_foreground_miou':reference['foreground_miou'],
        'confusion_matrix_exactly_matches_selection':summary['confusion_matrix']==reference['confusion_matrix'],
        'warning':'validation-only analysis of a validation-selected model; not independent test accuracy'})
    for filename,data in [('per-image.csv',rows),('ranked-images.csv',ranked)]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save_json(out/'summary.json',summary)
    save_json(EVIDENCE/f'validation-analysis-{args.model}.json',{'folder':str(out.relative_to(ROOT)),**summary})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['baseline','attention'],required=True)
    parser.add_argument('--segment-cap',type=int,default=5)
    main(parser.parse_args())
