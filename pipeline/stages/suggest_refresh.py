"""One entry point that refreshes everything the gallery and the review console show about the unsubmitted iNat photos.

Order (each step needs the one before; the run STOPS at the first failure and exits 1 naming it, never carries on with stale inputs):
  1. fetch_inat_live.py       the owner's live iNat observations (public read API) -> inat_observations.json
  2. inat_photos_fetch.py     the photos of newly labelled observations
  3. embed_index.py --backend bioclip2        their BioCLIP 2 vectors (only the missing ones)
  4. embed_loo.py --backend bioclip2          leave-one-out bins for the neighbour vote (suggest_bins.json)
  5. taxon_zeroshot.py calibrate              zero-shot bins; scores the observations posted since the last run on their own and prints DRIFT lines
                                              where the earlier calibration was optimistic (zeroshot_bins.json, "recent")
  6. per queue: inat_suggest.py, then `bio inat-gallery` (the queue's own prefix and manifest)

Config (<BIO_DATA>/suggest_refresh.json, or the path in BIO_SUGGEST_REFRESH): python_gpu, python_ops, stages_dir, package_src, hf_home, embed_config,
queues [{tag, prefix}]. Nothing is installed or registered by this script; a scheduled task that calls it is the owner's to create.
A lock file stops two runs overlapping (a lock older than LOCK_STALE_S is taken over). Log: <BIO_DATA>/suggest_refresh.log; last result:
<BIO_DATA>/suggest_refresh_last.json.
usage: suggest_refresh.py [--dry-run]     (--dry-run prints the steps and runs nothing)
"""
import json
import os
import subprocess
import sys
import time

DATA = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
LOCK_STALE_S = 6 * 3600


def load_config(path=None):
    path = path or os.environ.get('BIO_SUGGEST_REFRESH') or os.path.join(DATA, 'suggest_refresh.json')
    cfg = json.load(open(path, encoding='utf-8'))
    missing = [k for k in ('python_gpu', 'python_ops', 'stages_dir', 'package_src', 'hf_home', 'embed_config', 'queues') if k not in cfg]
    if missing:
        raise SystemExit(f'suggest_refresh: {path} lacks {", ".join(missing)}')
    return cfg


def build_steps(cfg, data=DATA):
    """[{name, argv, env}]: the steps in order; env holds only what differs from the caller's environment."""
    st = cfg['stages_dir']
    gpu_env = {'HF_HOME': cfg['hf_home'], 'HF_HUB_OFFLINE': '1', 'EMBED_CONFIG': cfg['embed_config'], 'PYTHONUNBUFFERED': '1'}
    ops_env = {'PYTHONPATH': cfg['package_src'], 'PYTHONUNBUFFERED': '1'}
    py, ops = cfg['python_gpu'], cfg['python_ops']
    steps = [
        {'name': 'fetch-live-observations', 'argv': [ops, os.path.join(data, 'fetch_inat_live.py')], 'env': ops_env},
        {'name': 'fetch-photos', 'argv': [py, os.path.join(st, 'inat_photos_fetch.py')], 'env': gpu_env},
        {'name': 'embed-photos', 'argv': [py, os.path.join(st, 'embed_index.py'), '--backend', 'bioclip2'], 'env': gpu_env},
        {'name': 'neighbour-bins', 'argv': [py, os.path.join(st, 'embed_loo.py'), '--backend', 'bioclip2'], 'env': gpu_env},
        {'name': 'zero-shot-calibrate', 'argv': [py, os.path.join(st, 'taxon_zeroshot.py'), 'calibrate'], 'env': gpu_env},
    ]
    for q in cfg['queues']:
        tag, prefix = q['tag'], q['prefix']
        steps.append({'name': f'suggest{tag}', 'argv': [py, os.path.join(st, 'inat_suggest.py')], 'env': {**gpu_env, 'BIO_QUEUE_TAG': tag}})
        steps.append({'name': f'gallery{tag}', 'argv': [ops, '-m', 'bio_automation.cli', 'inat-gallery',
                                                      '--queue', os.path.join(data, f'inat_queue{tag}.json'),
                                                      '--suggestions', os.path.join(data, f'inat_suggestions{tag}.json'), '--prefix', prefix],
                      'env': ops_env})
    return steps


class LockError(RuntimeError):
    pass


def acquire_lock(path, now=time.time, pid=os.getpid):
    """Create the lock file, or raise LockError when another run holds it. A lock older than LOCK_STALE_S is a crashed run's: it is replaced."""
    try:
        age = max(0.0, now() - os.path.getmtime(path))               # a file stamped a moment in the future (timestamp granularity, clock skew) is 0 min old, not -0
    except FileNotFoundError:
        age = None
    if age is not None:
        if age < LOCK_STALE_S:
            raise LockError(f'{path} is held ({age / 60:.0f} min old): another refresh is running (or crashed within {LOCK_STALE_S // 3600} h)')
        os.remove(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        f.write(f'{pid()} {time.strftime("%Y-%m-%dT%H:%M:%S")}\n')


def run_steps(steps, runner, clock=time.time, log=print):
    """Run the steps in order with runner(argv, env) -> exit code; stop at the first non-zero. Returns {ok, failed, steps: [{name, rc, seconds}]}."""
    done = []
    for s in steps:
        t0 = clock()
        log(f'== {s["name"]}: {" ".join(s["argv"])}')
        rc = runner(s['argv'], s['env'])
        done.append({'name': s['name'], 'rc': rc, 'seconds': round(clock() - t0, 1)})
        if rc != 0:
            log(f'== {s["name"]} FAILED with exit code {rc}; the remaining steps are NOT run')
            return {'ok': False, 'failed': s['name'], 'steps': done}
    return {'ok': True, 'failed': None, 'steps': done}


def subprocess_runner(argv, env):
    return subprocess.call(argv, env={**os.environ, **env})


def main():
    cfg = load_config()
    steps = build_steps(cfg)
    if '--dry-run' in sys.argv:
        for s in steps:
            print(s['name'], ' '.join(s['argv']))
        return 0
    lock = os.path.join(DATA, 'suggest_refresh.lock')
    try:
        acquire_lock(lock)
    except LockError as e:
        print(f'suggest_refresh: {e}', flush=True)
        return 1
    log_path = os.path.join(DATA, 'suggest_refresh.log')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    try:
        result = run_steps(steps, subprocess_runner, log=log)
        result['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        tmp = os.path.join(DATA, 'suggest_refresh_last.json.tmp')
        json.dump(result, open(tmp, 'w', encoding='utf-8'), indent=1)
        os.replace(tmp, os.path.join(DATA, 'suggest_refresh_last.json'))
        log('refresh ' + ('OK' if result['ok'] else f'FAILED at {result["failed"]}'))
        return 0 if result['ok'] else 1
    finally:
        os.remove(lock)


if __name__ == '__main__':
    sys.exit(main())
