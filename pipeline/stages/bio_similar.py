"""Find similar photos / organisms in the local index.

usage:
  bio_similar.py --image PATH [--k 10]          similar to an image file
  bio_similar.py --inat PHOTO_ID [--k 10]       similar to an indexed iNat photo
  bio_similar.py --frame LIGHTCRAFT_ID [--k 10] similar to an indexed library frame
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_index import CFG, Embedder, point_id
from qdrant_client import QdrantClient


def main():
    a = sys.argv[1:]
    k = int(a[a.index('--k') + 1]) if '--k' in a else 10
    qc = QdrantClient(path=CFG['qdrant_path'])
    if '--image' in a:
        vec = Embedder().embed([a[a.index('--image') + 1]])[0].tolist()
        exclude = None
    else:
        src, key = ('inat_photo', a[a.index('--inat') + 1]) if '--inat' in a else ('library_render', a[a.index('--frame') + 1])
        pid = point_id(src, key)
        got = qc.retrieve(CFG['collection'], [pid], with_vectors=True)
        if not got:
            sys.exit(f'{src}:{key} is not indexed')
        vec, exclude = got[0].vector, pid
    hits = qc.query_points(CFG['collection'], query=vec, limit=k + 1, with_payload=True).points
    for h in hits:
        if h.id == exclude:
            continue
        p = h.payload
        label = f"{p.get('common') or ''} ({p.get('taxon')})" if p.get('taxon') else f"frame {p.get('lightcraft_id')}"
        print(f"{h.score:.3f}  {p['source']:15s} {p['key']:>12s}  {label}  {p.get('quality') or ''} obs={p.get('obs_id') or ''}")


if __name__ == '__main__':
    main()
