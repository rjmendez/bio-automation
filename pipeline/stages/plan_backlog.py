"""Decide which frames to process: ONLY days the owner has not been through, resuming after where the owner stopped.

From day_ledger.py (exports, iNat observations, Lightroom work per day) and lrcat_signals.json (Pick flags):
  worked day        any exported JPEG, or >= 3 iNat observations, or Lightroom adjustments/ratings/labels/collections/keywords.
                    Frames of a worked day that were not exported were passed over on purpose: NOT processed.
  watermark         the latest worked day (the owner worked through the backlog in date order).
  backlog           untouched days AFTER the watermark -> every frame without a JPEG. Exception: a day where the owner already
                    started flagging Picks (Lightroom) but has not exported: only the Picks (that is the owner's own selection).
  gap day           untouched days BEFORE the watermark: processed too (literally not gone through) but kept in a separate folder.
  excluded          days already handled by another workflow (EXCLUDE_DAYS: 2026-09-27 has crops/enhanced + brooklyn_markings manifests).
Output: triage_targets.json {frame id: {day, kind, reason}} + printed summary.
"""
import collections
import json
import os

ROOT = r'F:\bio-automation-data'
EXCLUDE_DAYS = {'2026-09-27': 'already processed by the earlier Brooklyn-markings work (processed/crops + enhanced for all 63)'}
MIN_OBS_WORKED = 3


def main():
    led = json.load(open(os.path.join(ROOT, 'day_ledger.json')))
    idx = json.load(open(os.path.join(ROOT, 'triage_index.json')))
    sig = json.load(open(os.path.join(ROOT, 'lrcat_signals.json')))
    worked = {}
    for d, v in led.items():
        user_jpeg = v.get('F_jpeg', 0) + v.get('NAS_jpeg', 0)
        worked[d] = bool(user_jpeg or v.get('inat_obs', 0) >= MIN_OBS_WORKED or v.get('lr_adjusted') or v.get('lr_rated')
                         or v.get('lr_label') or v.get('lr_collection') or v.get('lr_keywords'))
    # days with only a handful of observations: the shoot was not worked through (e.g. 2026-08-01: 1 observation of 464 frames)
    fdays = sorted({i['day'] for i in idx})
    worked_f = [d for d in fdays if worked.get(d) and d not in EXCLUDE_DAYS]
    watermark = max(worked_f)
    targets, summary = {}, collections.OrderedDict()
    for it in idx:
        d = it['day']
        if d in EXCLUDE_DAYS:
            summary.setdefault(d, collections.Counter())['excluded'] += 1
            continue
        if worked.get(d):
            summary.setdefault(d, collections.Counter())['worked day, left as is'] += 1
            continue
        picks = [k for k, s in sig.items() if 'pick' in s]
        if d > watermark:
            if led.get(d, {}).get('lr_pick'):
                if str(it['id']) in sig and 'pick' in sig[str(it['id'])]:
                    targets[it['id']] = {'day': d, 'kind': 'backlog', 'reason': 'Pick-flagged by owner, not exported'}
                    summary.setdefault(d, collections.Counter())['process (your Picks)'] += 1
                else:
                    summary.setdefault(d, collections.Counter())['not picked, left as is'] += 1
            else:
                targets[it['id']] = {'day': d, 'kind': 'backlog', 'reason': 'untouched day after the last worked day'}
                summary.setdefault(d, collections.Counter())['process'] += 1
        else:
            targets[it['id']] = {'day': d, 'kind': 'gap', 'reason': 'untouched day before the last worked day'}
            summary.setdefault(d, collections.Counter())['process (gap day)'] += 1
    json.dump({str(k): v for k, v in targets.items()}, open(os.path.join(ROOT, 'triage_targets.json'), 'w'))
    print('last worked day (watermark):', watermark)
    print(f'{"day":10s} frames  decision')
    for d in sorted(summary):
        s = summary[d]
        if set(s) == {'worked day, left as is'}:
            continue
        print(f'{d:10s} {sum(s.values()):6d}  {dict(s)}')
    c = collections.Counter(v['kind'] for v in targets.values())
    left = sum(v.get('worked day, left as is', 0) for v in summary.values())
    print(f'TARGET frames: {len(targets)} ({dict(c)}) | frames on worked days left alone: {left} | of {len(idx)} with no JPEG')


if __name__ == '__main__':
    main()
