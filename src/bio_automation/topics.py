"""Queue topic names and their payload keys: the one definition every producer and consumer imports, so a handoff cannot drift by a typo
on one side. Queue.publish() checks every payload of a topic listed here against PAYLOAD_KEYS (check_payload) and refuses one that does
not carry exactly those keys."""

NAS_DAY_MATCHED = "nas.day_matched"            # {day}: a NAS gap day has DA-CLIP matches worth importing
LIBRARY_DAY_SELECTED = "library.day_selected"  # {day}: a day with iNat photos whose Stock source is a raw already in the LightCraft library
NAS_DAY_IMPORTED = "nas.day_imported"          # {day, frames}: that day's raws are in the LightCraft library; frames = path of the json list
                                               #   of imported frames ({lc_id, path, w, h, owner_edited} each, with real photo ids)
STOCK_RENDER_REQUESTED = "stock.render_requested"   # {day, csv, render_dir}: tagged Stock candidates ready for a retouched render into render_dir
STOCK_RENDER_IN_LIBRARY = "stock.render_in_library"   # {day, csv, render_dir}: a day with the owner's own edits (or unknown): render in the real library
STOCK_RENDERED = "stock.rendered"              # {day, csv, render_dir}: renders written, ready to label and publish; render_dir is the
                                               #   renders ROOT -- the day's files are render_dir/<year>/<day>/ (publish_candidates adds that)
GALLERY_REINDEX = "gallery.reindex"            # {gallery_dir, folders}: gallery folders (<year>/<year-month>) whose content changed

ALL = (NAS_DAY_MATCHED, LIBRARY_DAY_SELECTED, NAS_DAY_IMPORTED, STOCK_RENDER_REQUESTED, STOCK_RENDER_IN_LIBRARY, STOCK_RENDERED, GALLERY_REINDEX)

PAYLOAD_KEYS = {
    NAS_DAY_MATCHED: frozenset({"day"}),
    LIBRARY_DAY_SELECTED: frozenset({"day"}),
    NAS_DAY_IMPORTED: frozenset({"day", "frames"}),
    STOCK_RENDER_REQUESTED: frozenset({"day", "csv", "render_dir"}),
    STOCK_RENDER_IN_LIBRARY: frozenset({"day", "csv", "render_dir"}),
    STOCK_RENDERED: frozenset({"day", "csv", "render_dir"}),
    GALLERY_REINDEX: frozenset({"gallery_dir", "folders"}),
}


class TopicPayloadError(ValueError):
    """A payload for a known topic does not carry exactly that topic's keys."""


def check_payload(topic: str, payload: dict) -> None:
    """Raise TopicPayloadError unless `payload` has exactly PAYLOAD_KEYS[topic]; a topic not listed here is not checked."""
    want = PAYLOAD_KEYS.get(topic)
    if want is None:
        return
    have = set(payload)
    if have != want:
        missing, extra = sorted(want - have), sorted(have - want)
        raise TopicPayloadError(f"{topic} payload must carry exactly {sorted(want)}: missing {missing}, unexpected {extra}")
