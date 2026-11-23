"""Verify archive CRC/SHA, extract read-only source files, audit all triplets."""
import concurrent.futures as cf
import csv
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT, DATA, EVIDENCE, load_triplet, save_json

EXPECTED = {'train': 6815, 'val': 1333, 'test': 1929}

def extract():
    source = json.loads((EVIDENCE/'data-source.json').read_text())
    archive = ROOT/'data/raw/LEVIR-MCI-dataset.zip'
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    if h.hexdigest() != source['sha256']: raise ValueError('Archive changed')
    with zipfile.ZipFile(archive) as z:
        # Reading each member verifies its CRC; retain original index and RGB labels.
        for i, info in enumerate(z.infolist()):
            if info.is_dir(): continue
            target = (ROOT/'data'/info.filename).resolve()
            if not target.is_relative_to((ROOT/'data').resolve()):
                raise ValueError('Unsafe ZIP path')
            if not target.exists() or target.stat().st_size != info.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = z.read(info)
                target.write_bytes(payload)
                target.chmod(0o444)
            else:
                # Recheck CRC of existing files without allocating all files at once.
                import zlib
                if zlib.crc32(target.read_bytes()) != info.CRC: raise ValueError(f'CRC mismatch: {target}')
            if i % 5000 == 0: print('extracted', i, flush=True)

def inspect(item):
    split, name = item
    a, b, y = load_triplet(split, name)
    hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in (a, b)]
    row = {'split': split, 'filename': name,
           **{f'pixels_{i}': int((y == i).sum()) for i in range(3)},
           'a_pixel_sha256': hashes[0], 'b_pixel_sha256': hashes[1],
           'pair_sha256': hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
           'unordered_pair': ':'.join(sorted(hashes))}
    if split == 'train':
        values = np.concatenate([a.reshape(-1, 3), b.reshape(-1, 3)]).astype(np.float64)/255
        return row, values.sum(0), np.square(values).sum(0), len(values)
    return row, np.zeros(3), np.zeros(3), 0

def audit():
    started = time.monotonic()
    items = []
    for split, count in EXPECTED.items():
        sets = {d: {p.name for p in (DATA/split/d).glob('*.png')}
                for d in ['A', 'B', 'label', 'label_rgb']}
        if not all(s == sets['A'] for s in sets.values()): raise ValueError(f'Unpaired {split}')
        if len(sets['A']) != count: raise ValueError(f'Unexpected count {split}')
        items.extend((split, name) for name in sorted(sets['A']))
    rows, total, square, pixels = [], np.zeros(3), np.zeros(3), 0
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for i, (row, sm, sq, n) in enumerate(pool.map(inspect, items)):
            rows.append(row); total += sm; square += sq; pixels += n
            if i % 1000 == 0: print('audited', i, flush=True)
    with (EVIDENCE/'sample-manifest.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    mean = total/pixels
    stats = {'mean': mean.tolist(), 'std': np.sqrt(square/pixels-mean**2).tolist(),
             'source_split': 'train', 'images': 2*EXPECTED['train'], 'pixels_per_channel': pixels}
    save_json(EVIDENCE/'normalization.json', stats)
    summary = {}
    for split in EXPECTED:
        rs = [r for r in rows if r['split'] == split]
        counts = np.array([[r[f'pixels_{i}'] for i in range(3)] for r in rs])
        summary[split] = {'samples': len(rs), 'class_pixels': counts.sum(0).tolist(),
            'class_proportions': (counts.sum(0)/counts.sum()).tolist(),
            'images_with_road': int((counts[:,1]>0).sum()),
            'images_with_building': int((counts[:,2]>0).sum()),
            'images_without_target_change': int((counts[:,1:].sum(1)==0).sum())}
    duplicates = {}
    for key in ['pair_sha256', 'unordered_pair']:
        groups = defaultdict(list)
        for r in rows: groups[r[key]].append([r['split'], r['filename']])
        duplicates[key] = {'all_duplicate_groups': [g for g in groups.values() if len(g)>1],
            'cross_split_groups': [g for g in groups.values() if len({x[0] for x in g})>1]}
    individual = defaultdict(list)
    for r in rows:
        for t in ['a', 'b']: individual[r[f'{t}_pixel_sha256']].append([r['split'],r['filename'],t])
    duplicates['individual_cross_split'] = [g for g in individual.values() if len({x[0] for x in g})>1]
    save_json(EVIDENCE/'duplicates.json', duplicates)
    save_json(EVIDENCE/'data-audit.json', {'status': 'passed_image_integrity_and_pairing',
        'splits': summary, 'shape': [256,256,3], 'colors': [[0,0,0],[255,255,0],[255,0,0]],
        'cross_split_exact_pairs': len(duplicates['pair_sha256']['cross_split_groups']),
        'cross_split_unordered_pairs': len(duplicates['unordered_pair']['cross_split_groups']),
        'cross_split_individual_image_groups': len(duplicates['individual_cross_split']),
        'test_use': 'integrity, hashes and label counts only; no prediction, selection or tuning',
        'normalization': stats, 'elapsed_seconds': time.monotonic()-started})
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    extract()
    audit()
