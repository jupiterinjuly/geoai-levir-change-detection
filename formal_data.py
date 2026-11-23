"""Approved training list and train-only normalization for formal runs."""
import csv
import hashlib
import json
import numpy as np
from common import ROOT, EVIDENCE
from dataset import PairDataset

def approved_data():
    protocol = json.loads((ROOT/'configs/formal-approved-2026-11-20.json').read_text())
    manifest_path = ROOT/protocol['manifest']
    normalization_path = ROOT/protocol['normalization']
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != protocol['manifest_sha256']:
        raise ValueError('Approved manifest changed')
    if hashlib.sha256(normalization_path.read_bytes()).hexdigest() != protocol['normalization_sha256']:
        raise ValueError('Approved normalization changed')
    rows = list(csv.DictReader(manifest_path.open()))
    names = {s:[r['filename'] for r in rows if r['split']==s] for s in ['train','val','test']}
    if {s:len(x) for s,x in names.items()} != {'train':6794,'val':1333,'test':1929}:
        raise ValueError('Approved split sizes changed')
    train = [r for r in rows if r['split']=='train']
    held = {r[k] for r in rows if r['split']!='train' for k in ['a_pixel_sha256','b_pixel_sha256']}
    if any(r[k] in held for r in train for k in ['a_pixel_sha256','b_pixel_sha256']):
        raise ValueError('Exact train/heldout image overlap')
    stats = json.loads(normalization_path.read_text())
    return protocol,names,stats

class FormalDataset(PairDataset):
    def __init__(self,split,names,stats,seed=42,augment=False):
        if split not in ['train','val']:
            raise ValueError('Formal training does not read test samples')
        super().__init__(split,names,augment=augment,seed=seed)
        self.mean = np.array(stats['mean'],dtype=np.float32)
        self.std = np.array(stats['std'],dtype=np.float32)
