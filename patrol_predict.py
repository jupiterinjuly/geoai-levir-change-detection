"""Single-pair CPU inference with frozen exported checkpoint; no dataset reads."""
import base64
import io
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

from common import PALETTE
from models import UNet


def main():
    request = json.load(sys.stdin)
    torch.set_num_threads(1)
    checkpoint = torch.load(request['checkpoint'], map_location='cpu', weights_only=False)
    config = checkpoint['config']
    mean = np.array(config['normalization']['mean'], dtype=np.float32)
    std = np.array(config['normalization']['std'], dtype=np.float32)
    arrays = []
    for key in ('before', 'after'):
        with Image.open(io.BytesIO(base64.b64decode(request[key], validate=True))) as im:
            if im.size != (256, 256) or im.mode != 'RGB':
                raise ValueError('Expected 256x256 RGB image')
            arrays.append((np.asarray(im).astype(np.float32) / 255 - mean) / std)
    x = torch.from_numpy(np.concatenate(arrays, axis=2).transpose(2, 0, 1).copy()).unsqueeze(0)
    model = UNet(attention=request['model'] == 'attention')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model(x).argmax(1)[0].numpy().astype(np.uint8)
    seconds = time.perf_counter() - started
    output = io.BytesIO()
    Image.fromarray(PALETTE[prediction]).save(output, format='PNG')
    counts = np.bincount(prediction.ravel(), minlength=3)
    print(json.dumps({'mask_rgb': base64.b64encode(output.getvalue()).decode(),
                      'road_fraction': float(counts[1] / prediction.size),
                      'building_fraction': float(counts[2] / prediction.size),
                      'road_pixels': int(counts[1]), 'building_pixels': int(counts[2]),
                      'epoch': checkpoint['epoch'], 'cpu_forward_seconds': seconds,
                      'note': '无参考标签的单组 CPU 推理；不代表独立测试精度或标准化速度基准。'}))


if __name__ == '__main__':
    main()
