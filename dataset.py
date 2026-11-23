import csv
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from common import EVIDENCE, load_triplet, save_json

def manifest(split):
    with (EVIDENCE/'sample-manifest.csv').open() as f:
        return [r for r in csv.DictReader(f) if r['split'] == split]

def select_diagnostic():
    # Four strata, fixed seed; no test or validation image enters this diagnostic.
    rows = manifest('train')
    rng = random.Random(42)
    groups = {'road_only': [], 'building_only': [], 'both': [], 'empty': []}
    for row in rows:
        road, building = int(row['pixels_1'])>0, int(row['pixels_2'])>0
        key = 'both' if road and building else 'road_only' if road else 'building_only' if building else 'empty'
        groups[key].append(row['filename'])
    selected = []
    for key, count in [('road_only',4),('building_only',4),('both',6),('empty',2)]:
        if len(groups[key]) < count: raise ValueError(f'Insufficient {key}')
        selected.extend(rng.sample(sorted(groups[key]), count))
    save_json(EVIDENCE/'diagnostic-samples.json', {'split':'train','seed':42,
        'selection':'4 road-only, 4 building-only, 6 both, 2 empty; deliberately stratified, not representative',
        'filenames':selected})
    return selected

class PairDataset(Dataset):
    def __init__(self, split, filenames=None, augment=False, seed=42, cache=False):
        if split not in ['train','val','test']: raise ValueError(split)
        self.split = split
        self.filenames = filenames if filenames is not None else [r['filename'] for r in manifest(split)]
        self.augment, self.seed, self.epoch = augment, seed, 0
        stats = json.loads((EVIDENCE/'normalization.json').read_text())
        self.mean = np.array(stats['mean'], dtype=np.float32)
        self.std = np.array(stats['std'], dtype=np.float32)
        self.cache = {name:load_triplet(split,name) for name in self.filenames} if cache else {}

    def __len__(self): return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        a,b,y = self.cache[name] if self.cache else load_triplet(self.split,name)
        if self.augment:
            # Independent from model RNG/dropout; same seed/epoch/index -> same geometry.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            k, flip_x, flip_y = int(rng.integers(4)), bool(rng.integers(2)), bool(rng.integers(2))
            def transform(x):
                x = np.rot90(x,k)
                if flip_x: x = np.flip(x,1)
                if flip_y: x = np.flip(x,0)
                return x
            a,b,y = [transform(x) for x in (a,b,y)]
        x = np.concatenate([(a.astype(np.float32)/255-self.mean)/self.std,
                            (b.astype(np.float32)/255-self.mean)/self.std], -1)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))), torch.from_numpy(y.copy().astype(np.int64))
