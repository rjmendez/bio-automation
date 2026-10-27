"""Fetch the photos of the owner's labelled iNaturalist observations that the local photo cache does not have yet.

Why: the BioCLIP collection (embed_index.py --backend bioclip2) embeds the files of bioclip.sources.inat_photo; when the owner posts and identifies
new observations (inat_observations.json refreshed by fetch_inat_live.py) their photos must be here before they can be embedded, and before
taxon_zeroshot.py calibrate can test the suggestions on them.

Public CDN (static.inaturalist.org, the `medium` size of the URL the export lists, keeping its own extension), at most one photo per PACE_S, an
identifying User-Agent. A file is written to <name>.part and renamed, so a failed or interrupted download never leaves an empty file that a later
run would take for a cached photo. Exit status 1 when any download failed (the others are kept).
usage: inat_photos_fetch.py [--all] [--limit N]
  default: only observations newer than the last zero-shot calibration (zeroshot_bins.json max_obs_id); --all fetches every labelled observation's photos
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PACE_S = 0.2
USER_AGENT = os.environ.get('INAT_USER_AGENT', 'bio-automation/1.0 (+contact unset; read-only photo cache for local taxon suggestions)')
SIZES = re.compile(r'/(square|thumb|small|medium|large|original)\.')


def medium_url(url):
    """The same photo at the `medium` size (the extension is the original upload's and is kept)."""
    return SIZES.sub('/medium.', url, count=1)


def since_obs_id(bins_path):
    """The high-water mark of the last zero-shot calibration: the highest observation id it scored. SystemExit when there is none, so a first run can
    never silently pull the owner's whole photo history (21 k photos) and change every calibrated number."""
    if not os.path.exists(bins_path):
        raise SystemExit(f'inat_photos_fetch: {bins_path} does not exist; run taxon_zeroshot.py calibrate once to set the mark, or pass --all')
    mark = json.load(open(bins_path, encoding='utf-8')).get('max_obs_id')
    if not mark:
        raise SystemExit(f'inat_photos_fetch: {bins_path} has no max_obs_id; run taxon_zeroshot.py calibrate once to set it, or pass --all')
    return mark


def wanted(observations, folder, min_obs_id=0):
    """[(photo id, url)] of every photo of an observation with an id above *min_obs_id* that has a taxon and whose <id>.jpg is missing or empty in
    *folder*, in export order."""
    out = []
    for o in observations:
        if o['id'] <= min_obs_id or not (o.get('taxon') or {}).get('name'):
            continue
        for p in o.get('photos', []):
            f = os.path.join(folder, f"{p['id']}.jpg")
            if p.get('url') and not (os.path.isfile(f) and os.path.getsize(f) > 0):
                out.append((str(p['id']), medium_url(p['url'])))
    return out


def fetch_one(url, dst, get_bytes):
    """Download *url* to *dst* atomically; raises (and leaves no file at *dst*) when the download fails or returns nothing."""
    data = get_bytes(url)
    if not data:
        raise ValueError(f'{url}: empty response')
    tmp = dst + '.part'
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)                                           # a half-written .part must not stay behind either
        raise


def run(todo, folder, get_bytes, sleep=time.sleep, pace=PACE_S, log=print):
    """Fetch every (photo id, url) of *todo* into *folder*; returns (fetched, failed ids). One failure never stops the rest."""
    os.makedirs(folder, exist_ok=True)
    fetched, failed = 0, []
    for i, (pid, url) in enumerate(todo):
        try:
            fetch_one(url, os.path.join(folder, f'{pid}.jpg'), get_bytes)
            fetched += 1
        except Exception as e:
            failed.append(pid)
            log(f'FAIL {pid}: {type(e).__name__}: {str(e)[:100]}')
        if i < len(todo) - 1:
            sleep(pace)
        if (i + 1) % 200 == 0:
            log(f'  {i + 1}/{len(todo)} fetched {fetched} failed {len(failed)}')
    return fetched, failed


def http_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    return urllib.request.urlopen(req, timeout=30).read()


def main():
    from embed_index import CFG
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    folder = CFG['bioclip']['sources']['inat_photo']
    observations = json.load(open(CFG['inat_export'], encoding='utf-8'))['observations']
    mark = 0 if '--all' in sys.argv else since_obs_id(os.path.join(os.environ.get('BIO_DATA', 'F:/bio-automation-data'), 'zeroshot_bins.json'))
    todo = wanted(observations, folder, mark)
    todo = todo[:limit] if limit else todo
    print(f'{len(todo)} photos to fetch into {folder}', flush=True)
    fetched, failed = run(todo, folder, http_get)
    print(f'fetched {fetched}, failed {len(failed)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
