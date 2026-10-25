"""Embed cached photos with a local ONNX image encoder and index them in a local (embedded) Qdrant.

Config-driven (embed_config.json). Idempotent: points already indexed are skipped.
Payloads never contain coordinates or credentials: source, ids, taxon, quality, date, obscured flag only.
usage: embed_index.py [--limit N]

Execution providers (all optional keys in embed_config.json): "providers" (default CUDA then CPU), "cuda_device_id"
(overridden by env BIO_ORT_CUDA_DEVICE), "require_cuda" (true = refuse to run on CPU). A session that does not get CUDA is
reported at WARNING and counted (ort_providers.degraded_counts()), never silent. EMBED_CONFIG overrides the config path.
"""
import hashlib, json, os, sys, time
import numpy as np
from PIL import Image
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ort_providers import configured_device_id, open_session  # noqa: E402

CFG = json.load(open(os.environ.get('EMBED_CONFIG') or os.path.join(HERE, 'embed_config.json'), encoding='utf-8'))


class Embedder:
    def __init__(self):
        md = CFG['model_dir']
        pp = json.load(open(os.path.join(md, 'preprocessor_config.json'), encoding='utf-8'))
        size = pp.get('size', 224)
        self.size = size if isinstance(size, int) else size.get('height', 224)
        self.mean = np.array(pp.get('image_mean', [0.5, 0.5, 0.5]), dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(pp.get('image_std', [0.5, 0.5, 0.5]), dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.intra_op_num_threads = CFG.get('threads', 4)
        self.sess = open_session(ort, os.path.join(md, CFG['onnx_file']), sess_options=so, requested=CFG.get('providers'),
                                 device_id=configured_device_id(CFG.get('cuda_device_id')),
                                 require_cuda=bool(CFG.get('require_cuda', False)), label='embed_index')
        self.inp = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.out = CFG['output_name'] if CFG['output_name'] in outs else outs[-1]

    def prep(self, path):
        im = Image.open(path).convert('RGB').resize((self.size, self.size), Image.BICUBIC)
        a = (np.asarray(im, dtype=np.float32) / 255.0 - self.mean) / self.std
        return a.transpose(2, 0, 1)

    def embed(self, paths):
        x = np.stack([self.prep(p) for p in paths]).astype(np.float32)
        v = self.sess.run([self.out], {self.inp: x})[0]
        v = v.reshape(v.shape[0], -1)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def point_id(source, key):
    return int.from_bytes(hashlib.sha1(f'{source}:{key}'.encode()).digest()[:8], 'big') >> 1


def inat_meta():
    meta = {}
    d = json.load(open(CFG['inat_export'], encoding='utf-8'))['observations']
    for o in d:
        t = o.get('taxon') or {}
        for p in o.get('photos', []):
            meta[str(p['id'])] = {'obs_id': o['id'], 'observed_on': o.get('observed_on'), 'quality': o.get('quality'),
                                  'taxon': t.get('name'), 'common': t.get('common'), 'rank': t.get('rank'),
                                  'iconic': t.get('iconic'), 'obscured': bool(o.get('obscured'))}
    return meta


def backend_setup(name, cfg=None, embedder_cls=None, bioclip_cls=None):
    """(embedder, collection, sources, batch_size) for --backend *name*: 'siglip' (the ONNX index, default) or 'bioclip2' (cfg['bioclip'], GPU only)."""
    cfg = CFG if cfg is None else cfg
    if name == 'siglip':
        return (embedder_cls or Embedder)(), cfg['collection'], cfg['sources'], cfg['batch_size']
    if name == 'bioclip2':
        if bioclip_cls is None:
            from bioclip_embed import BioClipEmbedder as bioclip_cls
        bc = cfg['bioclip']
        return bioclip_cls(bc), bc['collection'], bc.get('sources') or cfg['sources'], bc.get('batch_size', cfg['batch_size'])
    raise SystemExit(f"embed_index: unknown --backend {name!r} (siglip or bioclip2)")


def main():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    backend = sys.argv[sys.argv.index('--backend') + 1] if '--backend' in sys.argv else 'siglip'
    emb, collection, sources, batch = backend_setup(backend)
    dim = emb.embed([next(os.path.join(sources[s], f) for s in sources
                          for f in os.listdir(sources[s]) if f.lower().endswith('.jpg'))]).shape[1]
    print(f'model ok: backend {backend}, {type(emb).__name__}, dim {dim}, collection {collection}', flush=True)
    qc = QdrantClient(path=CFG['qdrant_path'])
    if not qc.collection_exists(collection):
        qc.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    meta = inat_meta()
    total = 0
    t0 = time.time()
    for source, folder in sources.items():
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.jpg'))
        if limit:
            files = files[:limit]
        ids = [point_id(source, os.path.splitext(f)[0]) for f in files]
        have = set()
        for i in range(0, len(ids), 500):
            have |= {p.id for p in qc.retrieve(collection, ids[i:i + 500], with_payload=False, with_vectors=False)}
        todo = [(f, pid) for f, pid in zip(files, ids) if pid not in have]
        print(f'{source}: {len(files)} files, {len(have)} already indexed, {len(todo)} to embed', flush=True)
        B = batch
        for i in range(0, len(todo), B):
            chunk = todo[i:i + B]
            paths = [os.path.join(folder, f) for f, _ in chunk]
            try:
                vecs = emb.embed(paths)
            except Exception as e:
                print('  batch error', type(e).__name__, str(e)[:100], flush=True)
                continue
            pts = []
            for (f, pid), v in zip(chunk, vecs):
                key = os.path.splitext(f)[0]
                payload = {'source': source, 'key': key}
                if source == 'inat_photo':
                    payload.update(meta.get(key, {}))
                else:
                    payload['lightcraft_id'] = int(key) if key.isdigit() else key
                pts.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))
            qc.upsert(collection, pts)
            total += len(pts)
            if (i // B) % 20 == 0:
                print(f'  {source} {i + len(chunk)}/{len(todo)}  {total / max(time.time() - t0, 1):.1f} img/s', flush=True)
    print('done; points in collection:', qc.count(collection).count, f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
