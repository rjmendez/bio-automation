"""Second look at files flagged only for a person/body part (no face, no text/brand/property): is it ONLY the photographer's hands?

Owner rule 2026-10-12: the owner's own hands need no release. The first pass flags any person part and often names nothing specific.
This asks the same local vision model one narrow question per file. Positive only when the answer is explicit; anything else stays flagged.
Output hands_check.json {file: {only_hands: bool, other: str}} read by visionlib.hands_only. Usage: hands_check.py count | run
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_files as vf
import visionlib as L

OUT = os.path.join(vf.ROOT, 'hands_check.json')
EXPORT = r'F:\Bio-export\adobe-stock'
PERSON = {'person_visible', 'identifiable_person'}
Q = ('Look only for parts of a person in this photograph. Answer with JSON only, with exactly these keys: '
     '"hands_or_fingers" (true if a hand, fingers or a forearm is visible), '
     '"face_or_head" (true if any face, head, hair, ear or eye is visible), '
     '"other_body" (true if any other body part is visible: torso, leg, foot, shoulder, neck), '
     '"clothing_or_accessory" (true if clothing, a glove, a watch, a ring or jewellery is visible), '
     '"tattoo_or_mark" (true if a tattoo, scar or other identifying mark is visible on the skin), '
     '"shadow_of_person" (true if the shadow or reflection of a person is visible), '
     '"other_person_note" (at most 10 words about anything else of a person, or empty).')


def candidates():
    v = json.load(open(vf.OUT, encoding='utf-8'))
    out = []
    for k, e in v.items():
        t = L.tags_for(k, e)
        if not t:
            continue
        fl = [f for f in L.RELEASE_FIELDS if L.truthy(t.get(f))]
        if fl and set(fl) <= PERSON and not L.truthy(t.get('face_visible')) and not L.hands_only(t):
            p = next((os.path.join(EXPORT, d, k) for d in ('ready', '_needs_review', 'resized') if os.path.exists(os.path.join(EXPORT, d, k))), None)
            if p:
                out.append((k, p))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'count'
    todo = candidates()
    print('person-only flagged files needing a hands check:', len(todo), flush=True)
    if mode != 'run':
        return
    done = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    todo = [x for x in todo if x[0] not in done or 'error' in done[x[0]]]
    model = vf.pick_model()
    print('model:', model, 'to do', len(todo), flush=True)
    gpu = {'main_gpu': 1}
    t0 = time.time()
    for i, (k, p) in enumerate(todo, 1):
        try:
            body = {'model': model, 'stream': False, 'format': 'json', 'think': False, 'keep_alive': '30m', 'options': {'temperature': 0, **gpu},
                    'messages': [{'role': 'user', 'content': Q, 'images': [vf.small(p)]}]}
            try:
                r = vf.http('/api/chat', body)
            except Exception:
                body.pop('think', None)
                r = vf.http('/api/chat', body)
            a = L.flat(json.loads(r['message']['content']))
            only = (L.truthy(a.get('hands_or_fingers')) and not any(L.truthy(a.get(x)) for x in
                    ('face_or_head', 'other_body', 'clothing_or_accessory', 'tattoo_or_mark', 'shadow_of_person'))
                    and not str(a.get('other_person_note') or '').strip())
            done[k] = {'only_hands': bool(only), 'answer': a}
        except Exception as e:
            done[k] = {'error': f'{type(e).__name__}: {str(e)[:100]}'}
            if gpu and i < 5:
                gpu = {}
        if i % 20 == 0 or i == len(todo):
            json.dump(done, open(OUT, 'w'))
            el = time.time() - t0
            print(f'{i}/{len(todo)} {el / i:.1f}s per file eta {(len(todo) - i) * el / i / 60:.0f} min', flush=True)
    json.dump(done, open(OUT, 'w'))
    print('only hands:', sum(1 for v in done.values() if v.get('only_hands')), 'of', len(done), flush=True)


if __name__ == '__main__':
    main()
