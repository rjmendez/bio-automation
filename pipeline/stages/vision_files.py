"""One local-vision pass per exported JPEG copy: release screening (people, faces, text/logos, property) + descriptive tags.
Model chosen by capability (smallest vision-capable), never by name. Resumable; checkpoints to vision_files.json.
Conservative: any positive on a release field sends the file to _needs_review."""
import base64, io, json, logging, os, sys, time, urllib.error, urllib.request
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = r'F:\bio-automation-data'
OLLAMA = os.environ.get('BIO_OLLAMA', 'http://127.0.0.1:11440')
OUT = os.path.join(ROOT, 'vision_files.json')
DIRS = [r'F:\Bio-export\adobe-stock\ready', r'F:\Bio-export\adobe-stock\_needs_review', r'F:\Bio-export\adobe-stock\resized']

PROMPT = (
    'You review a nature photograph for stock-photo submission. Answer with JSON only, with exactly these keys. '
    'Descriptive: "subject_part" (e.g. flower, whole animal, fruiting body, leaf), "substrate" (e.g. wood, bark, leaf litter, soil, moss, grass), '
    '"habitat" (e.g. forest floor, meadow, garden, woodland, wetland), "colors" (up to 3 main colors of the subject), '
    '"viewpoint" (one of: close-up, macro, high-angle view, low-angle view, eye-level view, wide shot), '
    '"light" (one of: daylight, shade, backlit, overcast, flash, low light), "background" (short phrase), '
    '"adjectives" (up to 4 plain visual adjectives). '
    'Release screening (be strict, answer true if there is any doubt): '
    '"person_visible" (true if any person or any part of a person is visible, including a hand, finger, arm, leg, hair, clothing or a clear shadow of a person), '
    '"face_visible" (true if any face or part of a face is visible), '
    '"identifiable_person" (true if a person could be recognised from the image), '
    '"text_or_logo_visible" (true if any readable text, number, logo, label or sign is visible), '
    '"brand_or_product_visible" (true if a branded or manufactured product, vehicle, packaging, tool or clothing is visible), '
    '"property_identifiable" (true if a building, house, fence, license plate, artwork, mural or other identifiable private property or landmark is visible), '
    '"notes" (at most 12 words, what the release concern is, or empty). '
    'Do not name any species or brand.'
)


log = logging.getLogger('vision_files')
SLEEP = time.sleep
CHAT_RETRIES = 3
CHAT_BACKOFF_S = 2.0
RETRY_COUNT = 0               # retried /api/chat calls in this process (reported, never silent)


def gpu_options(environ=None):
    """The one place main_gpu is decided. No card is pinned by default: Ollama places the model by free VRAM. Set BIO_OLLAMA_MAIN_GPU
    to pin one; then every caller sends it on every request (a request without it makes Ollama reload the model onto another card)."""
    raw = (os.environ if environ is None else environ).get('BIO_OLLAMA_MAIN_GPU')
    if raw in (None, ''):
        return {}
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f'BIO_OLLAMA_MAIN_GPU must be an integer Ollama device index, got {raw!r}') from None
    if n < 0:
        raise ValueError(f'BIO_OLLAMA_MAIN_GPU must be >= 0, got {n}')
    return {'main_gpu': n}


def chat(body):
    """POST /api/chat. A model that rejects `think` is retried once without it; any other error is retried with the SAME options
    (the GPU pin included) after a backoff, each retry logged at WARNING and counted in RETRY_COUNT, then raised to the caller."""
    global RETRY_COUNT
    attempt = 0
    while True:
        try:
            return http('/api/chat', body)
        except urllib.error.HTTPError as e:
            if 'think' in body:
                body.pop('think')
                continue
            err = e
        except Exception as e:
            err = e
        if attempt >= CHAT_RETRIES:
            raise err
        RETRY_COUNT += 1
        log.warning('ollama /api/chat failed (%s: %s); retry %d of %d with options %s (total retries %d)',
                    type(err).__name__, str(err)[:100], attempt + 1, CHAT_RETRIES, body.get('options'), RETRY_COUNT)
        SLEEP(CHAT_BACKOFF_S * (2 ** attempt))
        attempt += 1


def http(path, body=None, timeout=300):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body else None, headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def pick_model():
    cands = []
    for m in http('/api/tags', timeout=30).get('models', []):
        try:
            caps = http('/api/show', {'model': m['name']}, timeout=30).get('capabilities', [])
        except Exception:
            continue
        if 'vision' in caps and 'embedding' not in caps:
            cands.append((m.get('size', 0), m['name']))
    cands.sort()
    return cands[0][1] if cands else None


def small(path, side=768):
    im = Image.open(path)
    im.draft('RGB', (side, side))          # fast partial JPEG decode
    im = im.convert('RGB')
    im.thumbnail((side, side))
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=85)
    return base64.b64encode(b.getvalue()).decode()


def ask(model, img, gpu_opts):
    body = {'model': model, 'stream': False, 'format': 'json', 'think': False, 'keep_alive': '30m',
            'options': {'temperature': 0, **gpu_opts}, 'messages': [{'role': 'user', 'content': PROMPT, 'images': [img]}]}
    r = chat(body)
    return json.loads(r['message']['content'])


def main():
    model = pick_model()
    print('vision model (smallest vision-capable):', model, flush=True)
    files = []
    for d in DIRS:
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith('.jpg')]
    done = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    todo = [f for f in files if os.path.basename(f) not in done or 'error' in done[os.path.basename(f)]]
    print(f'files {len(files)}, already done {len(files) - len(todo)}, to do {len(todo)}', flush=True)
    gpu = gpu_options()
    print('ollama options:', gpu, flush=True)
    t0 = time.time()
    for i, f in enumerate(todo, 1):
        name = os.path.basename(f)
        try:
            done[name] = {'tags': ask(model, small(f), gpu), 'model': model}
        except Exception as e:
            done[name] = {'error': f'{type(e).__name__}: {str(e)[:100]}'}
        if i % 25 == 0 or i == len(todo):
            json.dump(done, open(OUT, 'w'))
            el = time.time() - t0
            print(f'{i}/{len(todo)}  {el / i:.1f}s per file  eta {(len(todo) - i) * el / i / 60:.0f} min  errors {sum(1 for v in done.values() if "error" in v)}', flush=True)
    json.dump(done, open(OUT, 'w'))
    print('done', len(done), flush=True)


if __name__ == '__main__':
    main()
