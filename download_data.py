"""Pinned official archive; parallel HTTP ranges with resumable chunk ledger."""
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
import ssl
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
import certifi

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw'
META = ROOT / 'evidence'

def main():
    process_lock = (RAW/'download.lock').open('w')
    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = json.loads((META/'huggingface-dataset.json').read_text())['sha']
    entry = next(x for x in json.loads((META/'huggingface-files.json').read_text())
                 if x['path'] == 'LEVIR-MCI-dataset.zip')
    size, expected = entry['size'], entry['lfs']['oid']
    url = f'https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/resolve/{revision}/{entry["path"]}'
    target = RAW / entry['path']
    part = RAW / 'parallel-download.part'
    ledger = RAW / 'download-chunks.json'
    chunk = 16 * 1024**2
    done = set(json.loads(ledger.read_text())) if ledger.exists() else set()
    if done and (not part.exists() or part.stat().st_size != size):
        raise ValueError('Chunk ledger does not match partial archive')
    started = time.monotonic()
    lock = threading.Lock()
    context = ssl.create_default_context(cafile=certifi.where())
    if not target.exists():
        fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        def fetch(i):
            start, end = i*chunk, min((i+1)*chunk, size)-1
            for attempt in range(6):
                try:
                    chunk_file = RAW / f'chunk-{i}.tmp'
                    header_file = RAW / f'chunk-{i}.headers'
                    have = chunk_file.stat().st_size if chunk_file.exists() else 0
                    if have > end-start+1: raise ValueError('Oversize partial chunk')
                    if have < end-start+1:
                        tail = RAW / f'chunk-{i}.next'
                        result = subprocess.run(['curl', '-sS', '-L', '--fail', '--connect-timeout', '20',
                            '--max-time', '600', '--range', f'{start+have}-{end}',
                            '-D', str(header_file), '-o', str(tail),
                            url+f'?chunk={i}&attempt={attempt}&t={time.time_ns()}'])
                        if f'content-range: bytes {start+have}-{end}/{size}' not in header_file.read_text().lower():
                            raise ValueError('Invalid range response')
                        if tail.exists():
                            with chunk_file.open('ab') as f: f.write(tail.read_bytes())
                            tail.unlink()
                        if result.returncode: raise ValueError(f'curl interrupted, progress retained: {result.returncode}')
                    if chunk_file.stat().st_size != end-start+1: raise ValueError('Incomplete range')
                    with chunk_file.open('rb') as response:
                        offset = start
                        while block := response.read(1024**2):
                            if offset+len(block) > end+1: raise ValueError('Oversize range')
                            written = 0
                            while written < len(block):
                                written += os.pwrite(fd, block[written:], offset+written)
                            offset += len(block)
                        if offset != end+1: raise ValueError('Incomplete range')
                    with lock:
                        os.fsync(fd)
                        done.add(i)
                        temp = ledger.with_suffix('.tmp')
                        temp.write_text(json.dumps(sorted(done)))
                        temp.replace(ledger)
                        print(f'completed chunks {len(done)}/{(size+chunk-1)//chunk}; elapsed {time.monotonic()-started:.0f}s', flush=True)
                    chunk_file.unlink()
                    header_file.unlink()
                    return
                except Exception as exc:
                    print(f'chunk {i} attempt {attempt+1}: {exc}', flush=True)
                    if attempt == 5: raise
                    time.sleep(min(2**attempt, 15))
        try:
            with cf.ThreadPoolExecutor(max_workers=24) as pool:
                list(pool.map(fetch, [i for i in range((size+chunk-1)//chunk) if i not in done]))
        finally:
            os.close(fd)
        part.replace(target)
    h = hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    if target.stat().st_size != size or h.hexdigest() != expected:
        raise ValueError('Archive SHA256/size mismatch')
    (META/'data-source.json').write_text(json.dumps({
        'repository': 'https://huggingface.co/datasets/lcybuaa/LEVIR-MCI',
        'revision': revision, 'url': url, 'size_bytes': size,
        'sha256': h.hexdigest(), 'verified': True,
        'download_date_beijing': time.strftime('%Y-%m-%d', time.gmtime(time.time()+8*3600)),
    }, indent=2))
    print('Archive SHA256 verified', flush=True)

if __name__ == '__main__': main()
