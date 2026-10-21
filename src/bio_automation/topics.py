"""Queue topic names: the one definition every producer and consumer imports, so a handoff cannot drift by a typo on one side."""

NAS_DAY_MATCHED = "nas.day_matched"            # {day}: a NAS gap day has DA-CLIP matches worth importing
NAS_DAY_IMPORTED = "nas.day_imported"          # {day, ids_json}: that day's raws are in the LightCraft library, with real photo ids
STOCK_RENDER_REQUESTED = "stock.render_requested"   # {day, csv}: tagged Stock candidates ready for a retouched render
STOCK_RENDERED = "stock.rendered"              # {day, csv, render_dir}: renders written, ready to label and publish
GALLERY_REINDEX = "gallery.reindex"            # {gallery_dir, folders}: gallery folders whose content changed

ALL = (NAS_DAY_MATCHED, NAS_DAY_IMPORTED, STOCK_RENDER_REQUESTED, STOCK_RENDERED, GALLERY_REINDEX)
