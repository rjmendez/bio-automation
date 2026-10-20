"""Shared reader for vision_files.json entries.

The local vision model sometimes nests its answer ({"Descriptive": {...}, "Release screening": {...}}, any capitalisation) instead of
returning the flat schema in the prompt. Every consumer must read through flat(); reading top-level keys only makes nested entries
look "screened and clear" (that hid 286 release-flagged files on 2026-10-12).
"""
import json
import os
import re

RELEASE_FIELDS = ('person_visible', 'face_visible', 'identifiable_person', 'text_or_logo_visible', 'brand_or_product_visible',
                  'property_identifiable')
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
REDESCRIBED = os.path.join(ROOT, 'vision_redescribed.json')
DESC_FIELDS = ('subject_part', 'substrate', 'habitat', 'colors', 'viewpoint', 'light', 'background', 'adjectives')


def flat(tags):
    out = {}
    for k, v in (tags or {}).items():
        if isinstance(v, dict):
            out.update(flat(v))
        else:
            out[str(k).strip().lower().replace(' ', '_')] = v
    return out


def truthy(x):
    return x is True or (isinstance(x, str) and x.strip().lower() in ('true', 'yes'))


HAND = re.compile(r'\b(hands?|fingers?|thumbs?|palms?|fingertips?|knuckles?|forearms?|wrists?)\b', re.I)
NOT_HAND = re.compile(r'\b(face|faces|head|heads|torso|body|bodies|leg|legs|knee|feet|foot|shoe|shoes|boot|boots|person|people|man|woman|child|children|'
                      r'kid|selfie|portrait|tattoo|tattoos|ring|watch|jewelry|jewellery|hair|beard|shadow|shirt|clothing|clothes|jacket|glove|gloves|'
                      r'text|logo|sign|label|brand)\b', re.I)


def hands_only(t):
    """Owner rule (2026-10-12): the owner's own hands need no release. A person flag is hands-only when no face is visible and the model's
    own note/caption names a hand or fingers and names nothing else of a person. No note -> not hands-only (stays flagged)."""
    if truthy(t.get('face_visible')):
        return False
    text = ' '.join(str(t.get(f) or '') for f in ('notes', 'caption', 'notes_original'))
    return bool(HAND.search(text)) and not NOT_HAND.search(text)


def release_flags(tags, name=None):
    """Release fields that are positive. person_visible/identifiable_person caused only by the owner's hands are dropped."""
    t = tags_for(name, {'tags': tags}) if name else flat(tags)
    out = [k for k in RELEASE_FIELDS if truthy(t.get(k))]
    explicit = bool(name) and (hands_check().get(name) or {}).get('only_hands') is True   # second narrow look said: only hands
    if out and (explicit or hands_only(t)):
        out = [k for k in out if k not in ('person_visible', 'identifiable_person')]
    return out


_RD = None
_HC = None


def hands_check():
    global _HC
    if _HC is None:
        p = os.path.join(ROOT, 'hands_check.json')
        _HC = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    return _HC


def redescribed():
    global _RD
    if _RD is None:
        _RD = json.load(open(REDESCRIBED, encoding='utf-8')) if os.path.exists(REDESCRIBED) else {}
    return _RD


def tags_for(name, entry):
    """Flat tags for one file: release fields from the first (ID-blind) pass; descriptive fields and the release note replaced by the
    ID-aware re-description when one exists. Returns {} when the file was not screened."""
    t = flat((entry or {}).get('tags'))
    if not t:
        return {}
    r = (redescribed().get(name) or {}).get('tags')
    if r:
        r = flat(r)
        for f in DESC_FIELDS:
            if r.get(f) not in (None, '', []):
                t[f] = r[f]
        if r.get('caption'):
            t['caption'] = r['caption']
        t['id_plausible'] = r.get('id_plausible')
        if (redescribed().get(name) or {}).get('iconic') == 'Plantae' and 'fruiting body' in str(t.get('subject_part') or '').lower():
            t['subject_part'] = 'fruit'                  # fungal term on a plant
        t['notes_original'] = t.get('notes')
    return t
