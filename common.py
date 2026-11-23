"""Shared data semantics; test samples are never used for model development."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/LEVIR-MCI-dataset/images'
EVIDENCE = ROOT / os.environ.get('GEOAI_EVIDENCE_SUBDIR', 'evidence')
PALETTE = np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8)
NAMES = ['background', 'road_change', 'building_change']

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def decode_mask(rgb):
    rgb = np.asarray(rgb)
    matches = (rgb[..., None, :] == PALETTE).all(-1)
    if not matches.any(-1).all():
        unknown = np.unique(rgb[~matches.any(-1)], axis=0)
        raise ValueError(f'Unknown label colors: {unknown.tolist()}')
    return matches.argmax(-1).astype(np.uint8)

def load_triplet(split, filename):
    arrays = []
    for folder in ['A', 'B', 'label_rgb']:
        with Image.open(DATA/split/folder/filename) as im:
            if im.mode != 'RGB' or im.size != (256, 256):
                raise ValueError(f'Unexpected image {split}/{folder}/{filename}: {im.mode} {im.size}')
            im.load()
            arrays.append(np.array(im))
    return *arrays[:2], decode_mask(arrays[2])

def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.glob('*.py'))}

def confusion(y, prediction):
    return np.bincount((3*np.asarray(y).reshape(-1).astype(np.int64)
        + np.asarray(prediction).reshape(-1)), minlength=9).reshape(3, 3)

def metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp, actual, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    def divide(a, b):
        return [float(x/y) if y else None for x, y in zip(a, b)]
    iou = divide(tp, actual+predicted-tp)
    valid = [x for x in iou[1:] if x is not None]
    return {'iou': iou, 'precision': divide(tp, predicted), 'recall': divide(tp, actual),
            'f1': divide(2*tp, actual+predicted),
            'foreground_miou': float(np.mean(valid)) if valid else None,
            'foreground_valid_classes': len(valid), 'confusion_matrix': cm.astype(int).tolist()}
