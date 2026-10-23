"""Open-set zero-shot species suggestions: BioCLIP 2 image vectors scored against text vectors of EVERY species in iNaturalist's published taxonomy.

The kNN vote in inat_suggest.py can only name species the owner already posted. This names any species in the taxonomy (1.13 M in the 2026-10-01
export), with no reference photo. The class probability is softmax(logit_scale * cosine) over all species.

Source: iNaturalist's taxonomy DwC-A (https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip, taxa.csv), kept as an untrusted download in its own
folder on D:. Prompts follow BioCLIP's training text: ``a photo of <kingdom> <phylum> <class> <order> <family> <Genus species>.``

usage: taxon_zeroshot.py build <dwca.zip> <index dir> [--limit N]     (GPU; ~13 minutes for the whole taxonomy)
       taxon_zeroshot.py calibrate     (the bioclip.zeroshot_index of embed_config.json against the owner's labelled photos ->
                                        <BIO_DATA>/zeroshot_bins.json: accuracy of the pooled top-1 probability per observation)
"""
import csv
import io
import json
import os
import sys
import zipfile

import numpy as np

LINEAGE = ('kingdom', 'phylum', 'class', 'order', 'family')


def species_rows(zip_path, limit=None):
    """Yield {'id', 'name', 'kingdom'..'family', 'genus'} for every species row of the DwC-A's taxa.csv that has a kingdom and a binomial name."""
    with zipfile.ZipFile(zip_path) as z, z.open('taxa.csv') as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8', newline=''))
        n = 0
        for r in reader:
            if r['taxonRank'] != 'species' or not r['kingdom'] or not r['scientificName'] or ' ' not in r['scientificName']:
                continue
            yield {'id': int(r['id']), 'name': r['scientificName'], 'genus': r['genus'], **{k: r[k] for k in LINEAGE}}
            n += 1
            if limit and n >= limit:
                return


def prompt(row):
    return 'a photo of ' + ' '.join([row[k] for k in LINEAGE if row[k]] + [row['name']]) + '.'


def build_index(zip_path, out_dir, encode_text, limit=None, progress=None, chunk=50000):
    """Encode every species prompt with *encode_text* (list[str] -> float16 (N, dim), L2-normalised) and write species_text.npy, species.json, meta.json
    into *out_dir* (each atomically). Returns the number of species."""
    rows = list(species_rows(zip_path, limit))
    if not rows:
        raise ValueError(f'{zip_path}: no species rows found in taxa.csv')
    texts = [prompt(r) for r in rows]
    chunks = []
    for s in range(0, len(texts), chunk):
        chunks.append(encode_text(texts[s:s + chunk]))
        if progress:
            progress(min(s + chunk, len(texts)), len(texts))
    vecs = np.concatenate(chunks)
    if len(vecs) != len(rows):
        raise ValueError(f'encoder returned {len(vecs)} vectors for {len(rows)} species')
    os.makedirs(out_dir, exist_ok=True)
    npy_tmp = os.path.join(out_dir, 'species_text.tmp.npy')
    np.save(npy_tmp, vecs.astype(np.float16))
    os.replace(npy_tmp, os.path.join(out_dir, 'species_text.npy'))
    _write_json(os.path.join(out_dir, 'species.json'), [[r['id'], r['name'], r['kingdom'], r['phylum'], r['class'], r['order'], r['family'], r['genus']] for r in rows])
    _write_json(os.path.join(out_dir, 'meta.json'), {'source': os.path.basename(zip_path), 'source_bytes': os.path.getsize(zip_path),
                                                      'species': len(rows), 'dim': int(vecs.shape[1])})
    return len(rows)


def _write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class ZeroShot:
    """The species text index on a torch device. predict(image_vecs, scale) -> per image [(species row, probability)] of the top_k species."""

    def __init__(self, index_dir, torch_mod, device):
        self.torch, self.device = torch_mod, device
        self.species = json.load(open(os.path.join(index_dir, 'species.json'), encoding='utf-8'))
        vecs = np.load(os.path.join(index_dir, 'species_text.npy'))
        if len(vecs) != len(self.species):
            raise ValueError(f'{index_dir}: {len(vecs)} text vectors for {len(self.species)} species')
        self.text = torch_mod.from_numpy(vecs).to(device)

    def predict(self, image_vecs, scale, top_k=5, batch=64):
        t = self.torch
        out = []
        with t.no_grad():
            for s in range(0, len(image_vecs), batch):
                x = t.from_numpy(np.asarray(image_vecs[s:s + batch], dtype=np.float32)).to(self.device).to(self.text.dtype)
                logits = (x @ self.text.T).float() * scale
                lse = t.logsumexp(logits, dim=1, keepdim=True)
                vals, idx = t.topk(logits, top_k, dim=1)
                probs = (vals - lse).exp().cpu().numpy()
                for ii, pp in zip(idx.cpu().numpy(), probs):
                    out.append([(self.species[i], float(p)) for i, p in zip(ii, pp)])
        return out


EDGES = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))
ICONIC_CLASSES = frozenset({'Aves', 'Mammalia', 'Reptilia', 'Amphibia', 'Actinopterygii', 'Insecta', 'Arachnida'})


def iconic_of(row):
    """iNaturalist's iconic group of a species row [id, name, kingdom, phylum, class, ...]: the class for the seven animal classes iNat singles out,
    'Mollusca' for that phylum, 'Animalia' for other animals, else the kingdom (Plantae, Fungi, Protozoa, Chromista, ...)."""
    kingdom, phylum, klass = row[2], row[3], row[4]
    if kingdom != 'Animalia':
        return kingdom
    if klass in ICONIC_CLASSES:
        return klass
    return 'Mollusca' if phylum == 'Mollusca' else 'Animalia'


def pool_group(predictions, top=3):
    """One proposed observation = several photos. *predictions*: per photo, [(species row, probability)] (its top_k). Returns the *top* species by
    pooled probability = the sum of the photo probabilities for that species / the number of photos, as [(species row, pooled probability)]."""
    pooled, rows = {}, {}
    for photo in predictions:
        for row, p in photo:
            pooled[row[0]] = pooled.get(row[0], 0.0) + p
            rows[row[0]] = row
    ranked = sorted(pooled.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [(rows[i], v / len(predictions)) for i, v in ranked]


def observation_bins(per_photo, obs_ids, true_names, edges=EDGES):
    """Calibration of the POOLED top-1 probability, per observation (the queue's unit): [{lo, hi, n, accuracy}] where accuracy = share of observations
    whose pooled top-1 species name equals the observation's species. per_photo / obs_ids / true_names are parallel lists over photos."""
    groups = {}
    for preds, obs, name in zip(per_photo, obs_ids, true_names):
        g = groups.setdefault(obs, {'preds': [], 'name': name})
        g['preds'].append(preds)
    tops = []
    for g in groups.values():
        (row, p), = pool_group(g['preds'], top=1)
        tops.append((p, row[1] == g['name']))
    out = []
    for lo, hi in edges:
        hits = [h for p, h in tops if lo <= p < hi]
        out.append({'lo': lo, 'hi': hi, 'n': len(hits), 'accuracy': round(sum(hits) / len(hits), 3) if hits else None})
    return out


def measured(prob, bins):
    """Measured accuracy of a pooled probability: the bin it falls in; None when no bin holds it or the bin is empty."""
    for b in bins:
        if b['lo'] <= prob < b['hi']:
            return b['accuracy']
    return None


def calibrate(index_dir, collection_vectors, scale, torch_mod, device, out_path, top_k=5):
    """collection_vectors: [(vector, payload)] of the labelled iNat photos (species rank only are scored). Writes the observation-level bins JSON."""
    rows = [(v, p) for v, p in collection_vectors if p.get('rank') == 'species' and p.get('taxon')]
    zs = ZeroShot(index_dir, torch_mod, device)
    per_photo = zs.predict(np.array([v for v, _ in rows], dtype=np.float32), scale, top_k=top_k)
    bins = observation_bins(per_photo, [p['obs_id'] for _, p in rows], [p['taxon'] for _, p in rows])
    meta = json.load(open(os.path.join(index_dir, 'meta.json'), encoding='utf-8'))
    _write_json(out_path, {'index': meta, 'scale': scale, 'photos': len(rows), 'observations': sum(b['n'] for b in bins), 'bins': bins})
    return bins


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embed_index import CFG
    from bioclip_embed import BioClipEmbedder
    if cmd == 'build' and len(sys.argv) >= 4:
        limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
        emb = BioClipEmbedder(CFG['bioclip'])
        n = build_index(sys.argv[2], sys.argv[3], emb.encode_text, limit=limit,
                        progress=lambda done, total: print(f'  encoded {done}/{total}', flush=True))
        print(f'{n} species -> {sys.argv[3]} (logit scale {emb.logit_scale():.1f})', flush=True)
    elif cmd == 'calibrate':
        import torch
        from qdrant_client import QdrantClient
        emb = BioClipEmbedder(CFG['bioclip'])
        c = QdrantClient(path=CFG['qdrant_path'])
        pts, off = [], None
        while True:
            page, off = c.scroll(CFG['bioclip']['collection'], limit=2000, with_payload=True, with_vectors=True, offset=off)
            pts += [(p.vector, p.payload) for p in page if p.payload.get('source') == 'inat_photo']
            if off is None:
                break
        root = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
        out = os.path.join(root, 'zeroshot_bins.json')
        bins = calibrate(CFG['bioclip']['zeroshot_index'], pts, emb.logit_scale(), torch, emb.device, out)
        for b in bins:
            print(f'  pooled p {b["lo"]:.1f}-{min(b["hi"], 1):.1f}: n={b["n"]:4d}  correct {b["accuracy"]}', flush=True)
        print('->', out, flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
