"""Release/people screening for the iNat upload copies (they live outside the Adobe folders, so vision_files.py never saw them).

Same prompt and model choice as vision_files.py. Output vision_inat.json keyed by the copy's full path. Resumable.
The owner's own hands are not a concern (visionlib.hands_only); faces, other body parts, text, brands and property are shown to the
reviewer in the console as advisory flags.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_files as vf

ROOT = vf.ROOT
OUT = os.path.join(ROOT, 'vision_inat.json')


def main():
    q = json.load(open(os.path.join(ROOT, 'inat_queue.json')))
    paths = sorted({i['copy'] for i in q if i.get('copy') and os.path.exists(i['copy'])})
    done = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    todo = [p for p in paths if p not in done or 'error' in done[p]]
    model = vf.pick_model()
    print(f'copies {len(paths)}, to do {len(todo)}, model {model}', flush=True)
    gpu = {'main_gpu': 1}
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        try:
            done[p] = {'tags': vf.ask(model, vf.small(p), gpu), 'model': model}
        except Exception as e:
            done[p] = {'error': f'{type(e).__name__}: {str(e)[:100]}'}
            if gpu and i < 5:
                gpu = {}
        if i % 25 == 0 or i == len(todo):
            json.dump(done, open(OUT, 'w'))
            el = time.time() - t0
            print(f'{i}/{len(todo)} {el / i:.1f}s per file eta {(len(todo) - i) * el / i / 60:.0f} min', flush=True)
    json.dump(done, open(OUT, 'w'))
    print('done', len(done), flush=True)


if __name__ == '__main__':
    main()
