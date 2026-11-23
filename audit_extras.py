"""Reproduce label cross-checks and propose a conservative training-only cleanup.

Raw files and official validation/test lists are preserved. This script prepares
candidate lists; it never launches formal training or evaluates test predictions.
"""
import csv
import hashlib
import json
import time
from collections import Counter
import numpy as np
from PIL import Image
from common import EVIDENCE, DATA, load_triplet, decode_mask, save_json

def main():
    rows=list(csv.DictReader((EVIDENCE/'sample-manifest.csv').open()))
    colors,modes=Counter(),Counter();mismatches=[];started=time.monotonic()
    for r in rows:
        with Image.open(DATA/r['split']/'label'/r['filename']) as im:
            im.load();modes[im.mode]+=1;raw=np.array(im)
        if raw.shape!=(256,256,3) or not np.array_equal(raw[...,0],raw[...,1]) or not np.array_equal(raw[...,0],raw[...,2]):
            raise ValueError('Unexpected raw grayscale RGB label')
        values=raw[...,0];u,c=np.unique(values,return_counts=True)
        colors.update(dict(zip(u.tolist(),c.tolist())))
        if not np.isin(u,[0,128,255]).all():raise ValueError('Unknown raw label value')
        with Image.open(DATA/r['split']/'label_rgb'/r['filename']) as im:y=decode_mask(np.array(im))
        raw_y=np.zeros_like(values);raw_y[values==128]=1;raw_y[values==255]=2
        if not np.array_equal(raw_y,y):mismatches.append([r['split'],r['filename'],int((raw_y!=y).sum())])
    save_json(EVIDENCE/'raw-rgb-label-consistency.json',{'samples':len(rows),'grayscale_value_pixels':dict(colors),
        'modes':dict(modes),'mapping':{'0':'background','128':'road_change','255':'building_change'},
        'rgb_mapping':{'0,0,0':'background','255,255,0':'road_change','255,0,0':'building_change'},
        'mismatches':mismatches,'elapsed_seconds':time.monotonic()-started,
        'readme_discrepancy':'Archive readme label-rgb road color differs from actual pixels; class 128 consistently maps to yellow.'})
    if mismatches:raise ValueError('Raw and RGB annotations disagree: inspect before proceeding')
    duplicates=json.loads((EVIDENCE/'duplicates.json').read_text())['pair_sha256']
    confirmed=[];internal=[];reasons={}
    for group in duplicates['all_duplicate_groups']:
        s1,n1=group[0];s2,n2=group[1]
        a1,b1,y1=load_triplet(s1,n1);a2,b2,y2=load_triplet(s2,n2)
        assert np.array_equal(a1,a2) and np.array_equal(b1,b2)
        if s1!=s2:
            if (s1,s2)!=('train','test'):raise ValueError('Unexpected overlap policy requires review')
            confirmed.append({'train':n1,'test':n2,'a_equal':True,'b_equal':True,'labels_equal':bool(np.array_equal(y1,y2))})
            reasons[n1]='same image pair exists in official test'
        else:
            if s1!='train':raise ValueError('Unexpected within-heldout duplicate requires review')
            disagreement=int((y1!=y2).sum())
            internal.append({'ids':group,'label_disagreement_pixels':disagreement})
            names=sorted([n1,n2])
            if disagreement:
                for name in names:reasons[name]='identical training images with conflicting masks; exclude both without choosing a true annotation'
            else:reasons[names[1]]='identical training image pair and mask; retain smallest filename'
    clean=[r for r in rows if r['split']!='train' or r['filename'] not in reasons]
    with (EVIDENCE/'proposed-clean-manifest.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(clean[0]));writer.writeheader();writer.writerows(clean)
    train=[r for r in clean if r['split']=='train']
    assert len(train)==6794 and len({r['pair_sha256'] for r in train})==len(train)
    held={r[k] for r in clean if r['split']!='train' for k in ['a_pixel_sha256','b_pixel_sha256']}
    assert not any(r[k] in held for r in train for k in ['a_pixel_sha256','b_pixel_sha256'])
    full=json.loads((EVIDENCE/'normalization.json').read_text())
    n=full['pixels_per_channel'];mean=np.array(full['mean']);std=np.array(full['std'])
    total=mean*n;square=(std**2+mean**2)*n
    for name in reasons:
        a,b,_=load_triplet('train',name)
        for image in (a,b):
            values=image.reshape(-1,3).astype(np.float64)/255
            total-=values.sum(0);square-=np.square(values).sum(0);n-=len(values)
    mean=total/n
    save_json(EVIDENCE/'normalization-clean-train.json',{'mean':mean.tolist(),'std':np.sqrt(square/n-mean**2).tolist(),
        'source_split':'train','pairs':len(train),'images':2*len(train),'pixels_per_channel':n,
        'method':'subtract excluded training-image first/second moments from official train moments',
        'excluded':sorted(reasons),'status':'candidate; not used for the diagnostic runs'})
    save_json(EVIDENCE/'duplicates-bytewise-confirmed.json',confirmed)
    save_json(EVIDENCE/'within-train-duplicate-labels.json',internal)
    save_json(EVIDENCE/'proposed-training-exclusions.json',{'status':'proposal; awaits user confirmation before formal runs',
        'original_train_count':6815,'train_count_after_exclusion':len(train),'val_count':1333,'test_count':1929,
        'excluded_training_ids':sorted(reasons),'reasons':reasons,'raw_data_modified':False,
        'counts':{'cross_split_duplicate_training_pairs':5,'within_train_conflicting_pairs_excluded':10,'redundant_consistent_training_pairs_excluded':6},
        'test_predictions_inspected':False,'candidate_manifest_sha256':hashlib.sha256((EVIDENCE/'proposed-clean-manifest.csv').read_bytes()).hexdigest()})
    print('Candidate lists prepared:',len(train),'training pairs, excluded',len(reasons),flush=True)

if __name__=='__main__':main()
