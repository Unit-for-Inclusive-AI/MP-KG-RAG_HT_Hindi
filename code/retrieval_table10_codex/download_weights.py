"""Resume public pinned model weights over bounded, verified HTTPS ranges."""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE


def fetch_model(model, revision, filename):
    url = hf_hub_url(model, filename, revision=revision)
    meta = get_hf_file_metadata(url)
    cache = Path(HF_HUB_CACHE)/('models--' + model.replace('/', '--'))
    blob = cache/'blobs'/meta.etag
    if blob.exists() and blob.stat().st_size == meta.size:
        return
    prefixes = sorted(blob.parent.glob(meta.etag+'*.incomplete'), key=lambda p:p.stat().st_size, reverse=True)
    prefix = prefixes[0] if prefixes else None
    start = prefix.stat().st_size if prefix else 0
    folder = blob.parent/(meta.etag+'.segments')
    folder.mkdir(exist_ok=True)
    step = 16*1024*1024
    tasks = [(offset, min(offset+step, meta.size)-1) for offset in range(start, meta.size, step)]
    print(model, 'resume_bytes', start, 'total_bytes', meta.size, 'segments', len(tasks), flush=True)

    def fetch(bounds):
        lo, hi = bounds
        path = folder/f'{lo}-{hi}'
        if path.exists() and path.stat().st_size == hi-lo+1:
            return path
        for attempt in range(6):
            try:
                have = path.stat().st_size if path.exists() else 0
                offset = lo + have
                # Resolve afresh with the range to bind the CDN URL to this request.
                with requests.get(url, headers={'Range':f'bytes={offset}-{hi}'}, stream=True, timeout=(15,30)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {offset}-{hi}/{meta.size}':
                        raise RuntimeError('Server did not honor the requested byte range')
                    with path.open('ab') as f:
                        for chunk in r.iter_content(256*1024):
                            f.write(chunk)
                if path.stat().st_size != hi-lo+1:
                    raise RuntimeError('Short model segment')
                return path
            except Exception as exc:
                if attempt == 5:
                    raise
                print(model, 'retry', lo, type(exc).__name__, flush=True)
                time.sleep(1)

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(fetch,t) for t in tasks]
        for i, future in enumerate(as_completed(futures),1):
            future.result()
            if i%5 == 0 or i == len(tasks):
                print(model, 'segments_done', i, '/', len(tasks), flush=True)
    temporary = blob.with_suffix('.assembled')
    sha = hashlib.sha256()
    with temporary.open('wb') as out:
        paths = ([prefix] if prefix else []) + [folder/f'{lo}-{hi}' for lo,hi in tasks]
        for path in paths:
            with path.open('rb') as source:
                for block in iter(lambda:source.read(4*1024*1024), b''):
                    sha.update(block)
                    out.write(block)
    if temporary.stat().st_size != meta.size or sha.hexdigest() != meta.etag:
        raise RuntimeError('Model size or SHA-256 verification failed')
    os.replace(temporary,blob)
    link=cache/'snapshots'/revision/filename
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():
        link.symlink_to(Path('../../blobs')/meta.etag)
    print('Verified',model,sha.hexdigest(),flush=True)


if __name__ == '__main__':
    items=[('BAAI/bge-m3','5617a9f61b028005a4858fdac845db406aefb181','pytorch_model.bin'),
           ('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e','model.safetensors')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args:fetch_model(*args),items))
