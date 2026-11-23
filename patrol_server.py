"""Local validation/test browser and frozen-checkpoint CPU prediction."""
import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'application'
DATA = ROOT / 'data/LEVIR-MCI-dataset/images'
MODELS = ('baseline', 'attention')
CATALOG_LOCK = threading.Lock()
PREDICT_LOCK = threading.Lock()
CACHE = {}


def catalog(split="val"):
    if split not in ("val", "test"):raise ValueError("Unknown split")
    prefix="validation" if split=="val" else "test"
    count=1333 if split=="val" else 1929
    paths = [ROOT / f'evidence/{prefix}-analysis-{m}.json' for m in MODELS]
    signature = (split, *[p.stat().st_mtime_ns for p in paths])
    with CATALOG_LOCK:
        if CACHE.get('signature') == signature:
            return CACHE['public'], CACHE['private']
        public = {'split': split, 'source': 'LEVIR-MCI', 'models': {}, 'samples': []}
        private = {}
        samples = {}
        for m, p in zip(MODELS, paths):
            s = json.loads(p.read_text())
            if s['split'] != split or s['samples'] != count:
                raise ValueError('Only fully exported validation predictions are supported')
            digest = s['checkpoint_sha256']
            snapshot = APP / 'checkpoints' / f'{m}_{digest}.pt'
            if not snapshot.exists():
                candidates = [ROOT / f'runs/formal_{m}_seed42/best.pt']
                candidates += list((ROOT / 'archive').glob(f'**/formal_{m}_seed42/best.pt'))
                for candidate in candidates:
                    content = candidate.read_bytes()
                    if hashlib.sha256(content).hexdigest() == digest:
                        snapshot.parent.mkdir(parents=True, exist_ok=True)
                        temporary = snapshot.with_suffix('.tmp')
                        temporary.write_bytes(content)
                        temporary.replace(snapshot)
                        break
                else:
                    raise ValueError('Exported prediction checkpoint is unavailable')
            folder = ROOT / s['folder']
            private[m] = {'folder': folder, 'checkpoint': snapshot}
            public['models'][m] = {
                'epoch': s['checkpoint_epoch'], 'sha256': digest,
                'foreground_miou': s['foreground_miou'], 'iou': s['iou'],
                'road_area_mae_pp': s['road_area_mae_pp'],
                'building_area_mae_pp': s['building_area_mae_pp'],
                'coverage_top20': s['ranking_top20pct']['true_change_pixel_coverage'],
            }
            with (folder / 'per-image.csv').open() as f:
                for row in csv.DictReader(f):
                    name = row['filename']
                    if Path(name).name != name or not name.startswith(split+'_'):
                        raise ValueError('Unexpected validation sample name')
                    entry = samples.setdefault(name, {'name': name, 'group': row['size_group'], 'models': {}})
                    entry['models'][m] = {k: float(v) for k, v in row.items()
                                           if k not in ('filename', 'size_group')}
        if len(samples) != count or any(set(x['models']) != set(MODELS) for x in samples.values()):
            raise ValueError('Model validation sample lists do not match')
        public['samples'] = sorted(samples.values(), key=lambda x: x['name'])
        private['names'] = set(samples)
        CACHE.update(signature=signature, public=public, private=private)
        return public, private


def check_image(encoded):
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        if im.size != (256, 256) or im.mode != 'RGB' or im.format not in ('PNG', 'JPEG'):
            raise ValueError('请使用 256×256 的 RGB PNG 或 JPEG 影像。')
        im.load()
    return encoded


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, kind='application/json; charset=utf-8', status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            if url.path in ('/', '/patrol.css', '/patrol.js'):
                name = {'/': 'patrol.html', '/patrol.css': 'patrol.css', '/patrol.js': 'patrol.js'}[url.path]
                mime = {'patrol.html': 'text/html; charset=utf-8', 'patrol.css': 'text/css', 'patrol.js': 'text/javascript'}[name]
                return self.respond((APP / name).read_bytes(), mime)
            if url.path == '/favicon.ico':
                return self.respond(b'', 'image/x-icon', 204)
            query = parse_qs(url.query)
            split = query.get("split", ["val"])[0]
            if split not in ("val","test"):return self.respond({"error":"Unknown split"},status=404)
            public, private = catalog(split)
            if url.path == '/api/catalog':
                return self.respond(public)
            if url.path == '/api/image':
                query = parse_qs(url.query)
                name = query.get('sample', [''])[0]
                kind = query.get('kind', [''])[0]
                model = query.get('model', ['baseline'])[0]
                if name not in private['names'] or model not in MODELS:
                    return self.respond({'error': '未知验证样本或模型'}, status=404)
                if kind in ('before', 'after', 'truth'):
                    folder = {'before': 'A', 'after': 'B', 'truth': 'label_rgb'}[kind]
                    path = DATA / public['split'] / folder / name
                elif kind == 'prediction':
                    path = private[model]['folder'] / 'predictions_rgb' / name
                else:
                    return self.respond({'error': '未知图层'}, status=404)
                return self.respond(path.read_bytes(), 'image/png')
            self.respond({'error': '页面不存在'}, status=404)
        except Exception as error:
            self.log_error('%s', error)
            self.respond({'error': '读取失败，请查看本地运行记录。'}, status=500)

    def do_POST(self):
        if self.path != '/api/predict':
            return self.respond({'error': '页面不存在'}, status=404)
        # Only this local page may trigger the optional CPU prediction.
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://127.0.0.1:8791', 'http://localhost:8791'):
            return self.respond({'error': '来源不受支持'}, status=403)
        if not PREDICT_LOCK.acquire(blocking=False):
            return self.respond({'error': '已有一组影像正在预测，请稍后。'}, status=409)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4_000_000:
                raise ValueError('影像文件过大。')
            request = json.loads(self.rfile.read(length))
            model = request['model']
            if model not in MODELS:
                raise ValueError('未知模型。')
            check_image(request['before'])
            check_image(request['after'])
            public, private = catalog()
            if request.get('checkpoint_sha256') != public['models'][model]['sha256']:
                return self.respond({'error': '验证结果已更新，请刷新页面后重新预测。'}, status=409)
            request['checkpoint'] = str(private[model]['checkpoint'])
            result = subprocess.run([sys.executable, str(ROOT / 'patrol_predict.py')],
                                    input=json.dumps(request), text=True, capture_output=True,
                                    timeout=90, cwd=ROOT)
            if result.returncode:
                self.log_error('Prediction worker: %s', result.stderr[-2000:])
                raise RuntimeError('预测未完成，请查看本地运行记录。')
            output = json.loads(result.stdout)
            output['provenance'] = public['models'][model]
            self.respond(output)
        except (ValueError, KeyError, TypeError) as error:
            self.respond({'error': str(error)}, status=400)
        except Exception as error:
            self.log_error('%s', error)
            self.respond({'error': '预测未完成，请稍后重试。'}, status=500)
        finally:
            PREDICT_LOCK.release()


if __name__ == '__main__':
    catalog()
    print('巡查演示：http://127.0.0.1:8791；数据范围：验证集；上传预测：CPU', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8791), Handler).serve_forever()
