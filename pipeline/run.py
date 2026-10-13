"""bio pipeline: run the photo tagging / identification stages end to end and stop at the manual gates.

    python pipeline/run.py status            what exists, what is waiting on a human
    python pipeline/run.py plan              the stages that would run, with their preflight result
    python pipeline/run.py run [--stages a,b] [--force] [--dry-run]

Everything before the uploads is automatic and idempotent; the uploads to Adobe Stock and iNaturalist stay manual.
Paths are workstation defaults, overridable with BIO_DATA, BIO_EXPORT, BIO_SRC, BIO_NAS, BIO_PY.
Stdlib only. Nothing here uploads, deletes originals, or touches credentials: credential files are read by the stages
themselves and never printed.
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES_DIR = os.path.join(HERE, 'stages')
DATA = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
EXPORT = os.environ.get('BIO_EXPORT', r'F:\Bio-export\adobe-stock')
SRC = os.environ.get('BIO_SRC', r'F:\Bio')
NAS = os.environ.get('BIO_NAS', r'Y:\rjmendez-admin\DESKTOP-BVRDK4J\Data\F\Bio')
PY = os.environ.get('BIO_PY', os.path.join(DATA, 'venv', 'Scripts', 'python.exe'))
OLLAMA = os.environ.get('BIO_OLLAMA', 'http://127.0.0.1:11440')
LEDGER = os.path.join(DATA, 'pipeline_state.json')

# thresholds learned the hard way (2026-10-12): commit exhaustion aborted LightCraft three times in one day
MIN_COMMIT_HEADROOM_GB = 25.0
MAX_COMMIT_PCT = 85.0
MIN_DISK_FREE_GB = 30.0
MAX_ORPHAN_GIT = 5


def stage(script, *args, env=None, py=None):
    return {'cmd': [py or PY, os.path.join(STAGES_DIR, script), *args], 'env': env or {}}


STAGES = [
    {'name': 'inat_sync', 'desc': 'authenticated iNaturalist export, incl. obscured locations (private; stays in the data dir)',
     'cmd': ['wsl.exe', '-e', 'python3', os.path.join(STAGES_DIR, 'inat_sync.py').replace('\\', '/').replace('C:', '/mnt/c')],
     'env': {}, 'needs': ['wsl', 'creds:~/.inat'], 'long': False},
    {'name': 'nas_match', 'desc': 'match iNat photos to existing NAS JPEG exports by image correlation (read-only on the NAS)',
     **stage('nas_match.py'), 'needs': ['nas'], 'long': True},
    {'name': 'nas_stub', 'desc': 'convert NAS matches to the preview/writer formats', **stage('nas_stub.py'), 'needs': []},
    {'name': 'taxa', 'desc': 'taxonomy with English common names for matched taxa (public iNat API)',
     **stage('taxa_fetch.py', 'matches_stage1d.json', 'nas_matches_stub.json'), 'needs': ['net']},
    {'name': 'preview_local', 'desc': 'titles and keywords, buyer-first order (local library matches)',
     **stage('build_preview.py', 'matches_stage1d.json'), 'needs': []},
    {'name': 'preview_nas', 'desc': 'titles and keywords for NAS matches',
     **stage('build_preview.py', 'nas_matches_stub.json', env={'BIO_PREVIEW_PREFIX': 'adobe_preview_nas', 'BIO_ALLOW_JPEG': '1'}), 'needs': []},
    {'name': 'package_local', 'desc': 'metadata copies (IPTC+XMP, GPS stripped, size limits) from local exports',
     **stage('embed_meta.py'), 'needs': ['disk']},
    {'name': 'package_nas', 'desc': 'metadata copies from NAS exports',
     **stage('embed_meta.py', env={'BIO_SRC': NAS, 'BIO_MAP': 'export_map_nas.json', 'BIO_PREVIEW': 'adobe_preview_nas.csv',
                                   'BIO_MANIFEST': 'manifest_nas.csv', 'BIO_SKIP_EXISTING': '1'}), 'needs': ['nas', 'disk'], 'long': True},
    {'name': 'screen', 'desc': 'local vision model: release screening (people, faces, text, brands, property) + descriptive tags',
     **stage('vision_files.py'), 'needs': ['ollama', 'commit'], 'long': True},
    {'name': 'redescribe', 'desc': 'ID-aware re-description of photos whose vision text contradicts the iNat ID (fungus called an animal)',
     **stage('redescribe_vision.py', 'run'), 'needs': ['ollama'], 'long': True},
    {'name': 'hands_check', 'desc': 'second narrow look at person-only flags: owner hands need no release (explicit yes or stays flagged)',
     **stage('hands_check.py', 'run'), 'needs': ['ollama']},
    {'name': 'aggregate', 'desc': 'per-observation tags from per-file vision results', **stage('aggregate_vision.py'), 'needs': []},
    {'name': 'review', 'desc': 'release review page data (flagged photos) -> publish as a private page', **stage('make_review_page.py'),
     'needs': [], 'gate': 'RELEASE REVIEW: a human decides which flagged photos to drop'},
    {'name': 'inat_candidates', 'desc': 'prepared exports (local + NAS) that are NOT on iNaturalist yet, by image correlation per day',
     **stage('inat_candidates.py'), 'needs': ['nas', 'net'], 'long': True},
    {'name': 'gpx_index', 'desc': 'index the field phone GPX tracks (WSL ~/gpx) into one compact private file',
     'cmd': ['wsl.exe', '-e', 'python3', os.path.join(STAGES_DIR, 'gpx_index.py').replace('\\', '/').replace('C:', '/mnt/c')],
     'env': {}, 'needs': ['wsl'], 'long': False},
    {'name': 'geotag_raws', 'desc': 'location for raws without GPS from the phone track (stability-gated; no coverage = left blank)',
     **stage('geotag_new_raws.py'), 'needs': []},
    {'name': 'inat_queue', 'desc': 'iNat upload queue: location tiers, proposed observations, 2048 px upload copies with GPS',
     **stage('build_inat_queue.py'), 'needs': ['disk'], 'gate': 'INAT UPLOAD: you upload the copies and confirm the IDs'},
]
GATES = [
    ('release_review', 'Release review', 'Flagged photos (people, faces, text, brands, property). Only you can judge a release.'),
    ('adobe_upload', 'Adobe Stock upload', 'Upload the files in the ready folder through the Contributor Portal. Titles and keywords are already embedded.'),
    ('inat_upload', 'iNaturalist upload and ID confirmation', 'New photos are uploaded and identified on iNaturalist by you.'),
]


# ---------------------------------------------------------------- preflight
def commit_state():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    limit, avail = m.ullTotalPageFile / 2 ** 30, m.ullAvailPageFile / 2 ** 30      # commit limit and what is left of it
    return limit, limit - avail, avail


def orphan_git_count():
    try:
        out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq git.exe', '/FO', 'CSV', '/NH'], capture_output=True, text=True, timeout=20).stdout
        return sum(1 for line in out.splitlines() if line.lower().startswith('"git.exe"'))
    except Exception:
        return 0


def http_ok(url, timeout=5):
    try:
        urllib.request.urlopen(url, timeout=timeout).read(64)
        return True
    except Exception:
        return False


def check(need):
    """-> (ok, message)"""
    if need == 'commit':
        limit, used, avail = commit_state()
        pct = 100 * used / limit if limit else 0
        ok = avail >= MIN_COMMIT_HEADROOM_GB and pct <= MAX_COMMIT_PCT
        return ok, f'commit {used:.0f}/{limit:.0f} GB ({pct:.0f}%), {avail:.0f} GB headroom' + ('' if ok else f' (need >= {MIN_COMMIT_HEADROOM_GB:.0f} GB and <= {MAX_COMMIT_PCT:.0f}%)')
    if need == 'disk':
        os.makedirs(EXPORT, exist_ok=True)
        free = shutil.disk_usage(EXPORT).free / 2 ** 30
        return free >= MIN_DISK_FREE_GB, f'{free:.0f} GB free on the export drive' + ('' if free >= MIN_DISK_FREE_GB else f' (need >= {MIN_DISK_FREE_GB:.0f})')
    if need == 'nas':
        ok = os.path.isdir(NAS)
        return ok, 'NAS share mapped' if ok else f'NAS not reachable at {NAS} (map it first)'
    if need == 'ollama':
        ok = http_ok(OLLAMA + '/api/tags')
        return ok, 'Ollama reachable' if ok else f'Ollama not reachable at {OLLAMA}'
    if need == 'net':
        ok = http_ok('https://api.inaturalist.org/v1/taxa/1', 8)
        return ok, 'iNaturalist API reachable' if ok else 'iNaturalist API not reachable'
    if need == 'wsl':
        ok = shutil.which('wsl.exe') is not None
        return ok, 'wsl.exe present' if ok else 'wsl.exe not found'
    if need.startswith('creds:'):
        return True, 'credential file is read by the stage itself (never printed)'
    return True, ''


def preflight(st):
    results = [(n,) + check(n) for n in st.get('needs', [])]
    orphans = orphan_git_count()
    if orphans > MAX_ORPHAN_GIT:
        results.append(('git', False, f'{orphans} git.exe processes alive (orphaned listings hold commit memory; see notes)'))
    return results


# ---------------------------------------------------------------- ledger / status
def load_ledger():
    try:
        return json.load(open(LEDGER, encoding='utf-8'))
    except Exception:
        return {'stages': {}}


def save_ledger(led):
    os.makedirs(DATA, exist_ok=True)
    json.dump(led, open(LEDGER, 'w', encoding='utf-8'), indent=1)


def jload(name, default=None):
    try:
        return json.load(open(os.path.join(DATA, name), encoding='utf-8'))
    except Exception:
        return default


def count_jpg(folder):
    try:
        return sum(1 for f in os.listdir(folder) if f.lower().endswith('.jpg'))
    except Exception:
        return 0


def manifest_counts(name):
    import csv
    p = os.path.join(EXPORT, name)
    if not os.path.exists(p):
        return None
    c = {}
    for r in csv.DictReader(open(p, encoding='utf-8')):
        k = r['status'].split(':')[0]
        c[k] = c.get(k, 0) + 1
    return c


def cmd_status(_):
    led = load_ledger()
    obs = jload('inat_observations.json', {}).get('observations', [])
    vf = jload('vision_files.json', {})
    screened = sum(1 for v in vf.values() if v.get('tags'))
    TRUE = (True, 'true', 'True', 'yes')
    flagged = sum(1 for v in vf.values() if v.get('tags') and any(v['tags'].get(k) in TRUE for k in (
        'person_visible', 'face_visible', 'identifiable_person', 'text_or_logo_visible', 'brand_or_product_visible', 'property_identifiable')))
    nas = jload('nas_matches.json', {}).get('matches', [])
    ready, review, resized = (count_jpg(os.path.join(EXPORT, d)) for d in ('ready', '_needs_review', 'resized'))
    limit, used, avail = commit_state()
    print('=== data')
    print(f'  iNaturalist observations on disk : {len(obs)}')
    print(f'  NAS photo matches                : {len(nas)} ({len({m["obs"] for m in nas})} observations)')
    for name in ('manifest.csv', 'manifest_nas.csv', 'manifest_rendered.csv'):
        c = manifest_counts(name)
        if c is not None:
            print(f'  {name:26s}: {c}')
    print('=== package folders')
    print(f'  ready: {ready} | needs review: {review} | resized: {resized}')
    print(f'  vision screened: {screened} files | flagged for release review: {flagged}')
    unscreened = max(0, ready + review + resized - screened)
    print(f'  copies not yet screened (held out of ready): ~{unscreened}')
    print('=== machine')
    print(f'  commit {used:.0f}/{limit:.0f} GB ({100 * used / limit:.0f}%), {avail:.0f} GB headroom | git.exe: {orphan_git_count()}')
    print('=== last run per stage')
    for st in STAGES:
        r = led['stages'].get(st['name'])
        print(f'  {st["name"]:14s} {(r or {}).get("status", "never"):8s} {(r or {}).get("when", "")} {(r or {}).get("note", "")}')
    print('=== manual gates (these stay with you)')
    for key, title, why in GATES:
        extra = ''
        if key == 'release_review':
            extra = f' -> {flagged} flagged photos'
        if key == 'adobe_upload':
            extra = f' -> {ready} files in {os.path.join(EXPORT, "ready")}'
        print(f'  [{key}] {title}{extra}\n      {why}')


def cmd_plan(args):
    names = [s for s in (args.stages.split(',') if args.stages else [])]
    for st in STAGES:
        if names and st['name'] not in names:
            continue
        pf = preflight(st)
        bad = [f'{n}: {m}' for n, ok, m in pf if not ok]
        flag = 'BLOCKED' if bad else 'ok'
        print(f'{st["name"]:14s} [{flag}] {st["desc"]}')
        for b in bad:
            print(f'    - {b}')
        if st.get('gate'):
            print(f'    >> manual gate after this stage: {st["gate"]}')


def cmd_run(args):
    names = [s for s in (args.stages.split(',') if args.stages else [])]
    led = load_ledger()
    for st in STAGES:
        if names and st['name'] not in names:
            continue
        pf = preflight(st)
        bad = [f'{n}: {m}' for n, ok, m in pf if not ok]
        if bad and not args.force:
            print(f'[{st["name"]}] SKIPPED, preflight failed:')
            for b in bad:
                print('   -', b)
            led['stages'][st['name']] = {'status': 'blocked', 'when': time.strftime('%F %T'), 'note': bad[0][:80]}
            save_ledger(led)
            if st.get('long'):
                print('   (long stage: not retrying; fix the condition and re-run)')
            continue
        print(f'[{st["name"]}] {st["desc"]}')
        if args.dry_run:
            print('   would run:', ' '.join(st['cmd']))
            continue
        env = dict(os.environ, BIO_DATA=DATA, PYTHONIOENCODING='utf-8', **st.get('env', {}))
        t0 = time.time()
        p = subprocess.run(st['cmd'], env=env, cwd=DATA, capture_output=True, text=True, encoding='utf-8', errors='replace')
        tail = (p.stdout.strip().splitlines() or [''])[-1][:100]
        led['stages'][st['name']] = {'status': 'ok' if p.returncode == 0 else 'FAILED', 'when': time.strftime('%F %T'),
                                     'secs': round(time.time() - t0), 'note': tail if p.returncode == 0 else (p.stderr.strip().splitlines() or [''])[-1][:100]}
        save_ledger(led)
        print(f'   -> {led["stages"][st["name"]]["status"]} in {round(time.time() - t0)}s: {led["stages"][st["name"]]["note"]}')
        if p.returncode != 0:
            print('   stopping: a failed stage must be looked at before later stages run on its output')
            return 1
    print('\nmanual gates waiting:')
    for key, title, why in GATES:
        print(f'  - {title}: {why}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    for n in ('plan', 'run'):
        p = sub.add_parser(n)
        p.add_argument('--stages', default='')
        if n == 'run':
            p.add_argument('--force', action='store_true', help='run stages even if a preflight check fails')
            p.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    return {'status': cmd_status, 'plan': cmd_plan, 'run': cmd_run}[a.cmd](a) or 0


if __name__ == '__main__':
    sys.exit(main())
