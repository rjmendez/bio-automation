"""Release review page: thumbnails of every photo the local vision screen flagged, with the reason, sorted by severity.
Static HTML + inline thumbnails. Per-viewer checkboxes (localStorage) and a copy-list button; nothing leaves the page."""
import base64, csv, html, io, json, os, sys
import visionlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
OUTF = os.environ.get('BIO_REVIEW_PAGE', os.path.join(ROOT, 'release_review.html'))
SIDE, Q = int(os.environ.get('REVIEW_THUMB', 420)), int(os.environ.get('REVIEW_Q', 68))
FIELDS = [('identifiable_person', 'Identifiable person'), ('face_visible', 'Face visible'), ('person_visible', 'Person / body part'),
          ('text_or_logo_visible', 'Text or logo'), ('brand_or_product_visible', 'Brand or product'), ('property_identifiable', 'Identifiable property')]
SEV = {k: i for i, (k, _) in enumerate(FIELDS)}


def truthy(x):
    return x is True or (isinstance(x, str) and x.strip().lower() in ('true', 'yes'))


vf = json.load(open(os.path.join(ROOT, 'vision_files.json'), encoding='utf-8'))
rows = []
for mf in ('manifest.csv', 'manifest_nas.csv', 'manifest_rendered.csv'):
    p = os.path.join(os.environ.get('BIO_EXPORT', r'F:\Bio-export\adobe-stock'), mf)
    if os.path.exists(p):
        rows += [dict(r, _manifest=mf) for r in csv.DictReader(open(p, encoding='utf-8'))]
seen, cards = set(), []
for r in rows:
    src = r['source_jpeg']
    parts = src.replace('\\', '/').split('/')
    day = next((c for c in parts if len(c) == 10 and c[4] == '-' and c[7] == '-'), '')
    key = f'{day}_{os.path.basename(src)}'
    t = visionlib.tags_for(key, vf.get(key))
    if not t or key in seen:
        continue
    flags = [k for k, _ in FIELDS if k in visionlib.release_flags(vf.get(key, {}).get('tags'), key)]
    if not flags:
        continue
    seen.add(key)
    try:
        im = Image.open(src)
        im.draft('RGB', (SIDE * 2, SIDE * 2))
        im = im.convert('RGB')
        im.thumbnail((SIDE, SIDE))
        b = io.BytesIO()
        im.save(b, 'JPEG', quality=Q)
        thumb = 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()
    except Exception:
        thumb = ''
    cards.append({'key': key, 'title': r['title'], 'flags': flags, 'note': (t.get('caption') or t.get('notes') or '').strip(), 'thumb': thumb, 'day': day,
                  'src': os.path.basename(src), 'obs': r['obs_ids']})
cards.sort(key=lambda c: (min(SEV[f] for f in c['flags']), c['day'], c['src']))
label = dict(FIELDS)


def card(c):
    chips = ''.join(f'<span class="chip sev{min(SEV[f], 3)}">{html.escape(label[f])}</span>' for f in c['flags'])
    group = 'people' if any(f in ('identifiable_person', 'face_visible', 'person_visible') for f in c['flags']) else 'other'
    note = f'<p class="note">{html.escape(c["note"])}</p>' if c['note'] and 'no release' not in c['note'].lower() else ''
    img = f'<img loading="lazy" alt="{html.escape(c["title"])}" src="{c["thumb"]}">' if c['thumb'] else '<div class="noimg">no preview</div>'
    return (f'<article class="card" data-group="{group}" data-key="{html.escape(c["key"])}">{img}<div class="body">'
            f'<h3>{html.escape(c["title"])}</h3><p class="meta">{html.escape(c["src"])} <span>{html.escape(c["day"])}</span></p>'
            f'<div class="chips">{chips}</div>{note}'
            f'<label class="drop"><input type="checkbox" class="dropbox"> Drop from the submission set</label></div></article>')


n_people = sum(1 for c in cards if any(f in ('identifiable_person', 'face_visible', 'person_visible') for f in c['flags']))
page = f'''<title>Release Review</title>
<style>
/* layout: a header with counts and filters, then a responsive grid of photo cards, one decision per card */
:root {{ --bg:#f3f4f6; --panel:#fff; --fg:#14181f; --muted:#5a6472; --line:#d5d9e0; --accent:#0b6b73; --chip:#e9ecf1;
  --s0:#b3261e; --s1:#b4561b; --s2:#7a5c00; --s3:#4b5563; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#12151a; --panel:#1b2027; --fg:#e8ebf0; --muted:#9aa4b2; --line:#2e3540; --accent:#4cc3cc; --chip:#262d37;
  --s0:#ff8a80; --s1:#ffb27a; --s2:#e6c15c; --s3:#aab3c0; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg:#12151a; --panel:#1b2027; --fg:#e8ebf0; --muted:#9aa4b2; --line:#2e3540; --accent:#4cc3cc; --chip:#262d37;
  --s0:#ff8a80; --s1:#ffb27a; --s2:#e6c15c; --s3:#aab3c0; color-scheme: dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,'Segoe UI',sans-serif; line-height:1.45; padding-inline:max(16px,4vw); padding-block:20px 56px; }}
main {{ max-width:1100px; margin-inline:auto; display:flex; flex-direction:column; gap:16px; }}
h1 {{ font-size:clamp(1.4rem,3.2vw,2rem); margin:0 0 4px; }}
.lede {{ color:var(--muted); margin:0; max-width:70ch; }}
.bar {{ position:sticky; top:env(safe-area-inset-top,0px); background:var(--bg); padding-block:10px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; border-bottom:1px solid var(--line); z-index:2; }}
.bar button {{ background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:6px 12px; font:inherit; cursor:pointer; }}
.bar button[aria-pressed="true"] {{ border-color:var(--accent); color:var(--accent); font-weight:600; }}
.bar .count {{ margin-left:auto; color:var(--muted); font-size:.92rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; display:flex; flex-direction:column; }}
.card img, .noimg {{ width:100%; aspect-ratio:3/2; object-fit:cover; display:block; background:var(--chip); }}
.noimg {{ display:grid; place-items:center; color:var(--muted); }}
.body {{ padding:10px 12px 12px; display:flex; flex-direction:column; gap:6px; min-width:0; }}
.body h3 {{ font-size:.98rem; margin:0; overflow-wrap:anywhere; }}
.meta {{ margin:0; font-size:.78rem; color:var(--muted); font-family:ui-monospace,Consolas,monospace; overflow-wrap:anywhere; display:flex; gap:8px; flex-wrap:wrap; }}
.chips {{ display:flex; flex-wrap:wrap; gap:5px; }}
.chip {{ font-size:.76rem; padding:2px 8px; border-radius:4px; background:var(--chip); border:1px solid var(--line); }}
.sev0 {{ color:var(--s0); border-color:var(--s0); }} .sev1 {{ color:var(--s1); border-color:var(--s1); }} .sev2 {{ color:var(--s2); border-color:var(--s2); }} .sev3 {{ color:var(--s3); }}
.note {{ margin:0; font-size:.84rem; color:var(--muted); }}
.drop {{ display:flex; gap:8px; align-items:center; font-size:.9rem; padding-top:4px; border-top:1px solid var(--line); margin-top:2px; }}
.card.dropped {{ outline:2px solid var(--s0); }}
textarea {{ width:100%; min-height:7rem; background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:8px; font-family:ui-monospace,Consolas,monospace; font-size:.82rem; }}
.notice {{ background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:6px; padding:12px 14px; font-size:.92rem; }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>
<main>
<div><h1>Release review</h1>
<p class="lede">{len(cards)} photos flagged by a local vision model that screens for people, faces, text, brands and identifiable property. Tick the ones to drop from the Adobe Stock set. Everything unticked stays in.</p></div>
<div class="notice">This is a screen, not a clearance. The model errs towards flagging, so many of these will be a hand or a shadow. It can also miss small or partial faces, and it cannot judge whether a release exists. The decision is yours.</div>
<div class="bar" role="toolbar" aria-label="Filter">
<button type="button" data-f="all" aria-pressed="true">All ({len(cards)})</button>
<button type="button" data-f="people" aria-pressed="false">People ({n_people})</button>
<button type="button" data-f="other" aria-pressed="false">Text, brand, property ({len(cards) - n_people})</button>
<span class="count" id="count">0 marked to drop</span>
</div>
<div class="grid" id="grid">{''.join(card(c) for c in cards)}</div>
<section class="notice"><strong>Your list</strong>: copy this and send it back, and I will move those files out of the set.
<p><button type="button" id="copy">Copy list</button></p>
<textarea id="out" readonly aria-label="Files marked to drop" placeholder="Nothing marked yet."></textarea></section>
</main>
<script>
(function () {{
  var KEY = 'release-review-drop';
  var saved = {{}};
  try {{ saved = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
  var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));
  var out = document.getElementById('out'), count = document.getElementById('count');
  function render() {{
    var list = [];
    cards.forEach(function (c) {{
      var k = c.getAttribute('data-key'), box = c.querySelector('.dropbox');
      c.classList.toggle('dropped', box.checked);
      if (box.checked) list.push(k);
    }});
    count.textContent = list.length + ' marked to drop';
    out.value = list.join('\\n');
    try {{ var o = {{}}; list.forEach(function (k) {{ o[k] = 1; }}); localStorage.setItem(KEY, JSON.stringify(o)); }} catch (e) {{}}
  }}
  cards.forEach(function (c) {{
    var box = c.querySelector('.dropbox');
    box.checked = !!saved[c.getAttribute('data-key')];
    box.addEventListener('change', render);
  }});
  document.querySelectorAll('.bar [data-f]').forEach(function (b) {{
    b.addEventListener('click', function () {{
      var f = b.getAttribute('data-f');
      document.querySelectorAll('.bar [data-f]').forEach(function (x) {{ x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); }});
      cards.forEach(function (c) {{ c.hidden = !(f === 'all' || c.getAttribute('data-group') === f); }});
    }});
  }});
  document.getElementById('copy').addEventListener('click', function () {{
    out.focus(); out.select();
    try {{ navigator.clipboard.writeText(out.value).catch(function () {{ document.execCommand && document.execCommand('copy'); }}); }}
    catch (e) {{ try {{ document.execCommand('copy'); }} catch (e2) {{}} }}
  }});
  render();
}})();
</script>
'''
open(OUTF, 'w', encoding='utf-8').write(page)
print(f'cards {len(cards)} (people {n_people}) | page {len(page) / 1e6:.2f} MB -> {OUTF}')
