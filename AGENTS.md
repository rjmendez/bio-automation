# AGENTS.md

Instructions for coding agents working in this repository.

## Standing rules (owner decisions; do not re-ask, do not forget)

Ground first: search Loci (`investigation_search`, `rag_context_search`) before
starting work, then verify against live code and data.

**Before processing photos, check what is already on iNat.** Pull the owner's
live observations (public API, read-only, 1 req/s, identifying User-Agent) and
match photos to frames by analyzed similarity (DA-CLIP), using capture time only to
bracket which frames to compare (see "Stock labelling comes from iNat" below). Local snapshots go stale (`inat_observations.json` lacked
113 observations created after it was built). Never re-upload a frame that is on iNat.

**Export standards** (`render.py`): real raw decode to sRGB, never the embedded
preview. iNat: long edge 2048 px, JPEG quality 95, at most 20 MB per photo
(forum-confirmed, no official page). Adobe Stock: full sensor resolution, quality
95, 4-100 MP, at most 45 MB (Adobe helpx). A q90 or 2048 px file is not Stock-ready
(below the 4 MP floor). Re-encoding an existing JPEG never fixes quality: re-export
from the RAW.

**Where files go.** Photo working files and intermediates (enhanced frames, trial
output) go under `D:\bio-cache\photos`. Final renders stay on F:. Nothing heavy on
C:. The Topaz models and their extracted cache stay on C: where they are
(`MODELS_DIR`/`CACHE_DIR` in `topaz.py`); the photo cache is not the model cache.

**Enhancement.** `lce` (github.com/rjmendez/lce) is the LightCraft enhancement
plugin; its public registry is open weights only. The owner's Topaz models on C:
are used through `bio_automation.topaz` (OpenVINO), never committed or shipped in
`lce`. `enhance()` must keep the source EXIF and ICC profile. Always compare a
100% crop before and after on a real frame and measure; never assume a model helps.
Measured 2026-10-20 (hardening audit, 9 real frames): `slb` is Topaz *Sharpen
Refocus*, not Lens Blur; its `t_param1` is a second sharpening amplitude, not a
denoiser; the auto-param `lensblur` value is about 0.00-0.10 so default
auto-sharpen is a no-op; stronger settings add halos and invented texture. Do not
sharpen iNat 2048 px uploads. The real denoise models (`dnt_beta`, `isoa`, `isob`)
and detail models (`ghqv2`) are the ones to test, against synthetic ground truth.

**No CPU processing (owner, 2026-10-20).** Model inference runs on the GPU only
(onnxruntime CUDA); a missing CUDA/cuDNN setup must fail loudly, never fall back
to CPU. OpenVINO CPU inference and CPU denoise are not allowed.

**LightCraft batch work runs headless** (`BIO_LC_HEADLESS=1`, GUI closed, one
process on the library). Do not drive the live GUI session.

**iNat posting is human-approved per observation**; locations and taxa are chosen
by the owner in iNat. Build upload sets (files + CSV), never auto-post.

**Stock labelling comes from iNat, enforced in code (2026-10-20).** A frame is a Stock
candidate ONLY if it IS one of the photos the owner uploaded to a live iNat observation.
**Match photos by analyzed similarity; timestamps only bracket (owner rule, 2026-10-20).**
Each iNat photo takes the frame that looks most like it (DA-CLIP content vectors on the GPU,
argmax, >= 0.97), searching only frames captured within +-15 min of the observation
(`inat_stock.match_photos_to_frames`). Never match by timestamp alone: a 90 s time match used
two frames that were never uploaded and missed three that were (observation 409468156) and put
a species on frames shot near an unrelated observation. Every uploaded photo that passes the
gates is used; unuploaded sibling shots are not. Each matched observation is labelled from its
live-fetched iNat data (`fetch_inat_observation` -> `stock.build_metadata` -> title, 15-50 keywords,
caption, Adobe category number). Keywords include the family/order/class names and common names from iNat's taxonomy
(`fetch_taxon_lineage`, also stored by `bio fetch-inat`). Frames that are not an uploaded photo
are never candidates; the code says why (`not one of the photos uploaded to an iNat observation`). `bio stock-tag-queue` REQUIRES
`--observations <live snapshot, at most 3 days old>`; `bio process` ends with
`tag_observation`, which needs the folder to be fetch-inat'd. `stock-publish` writes the
label INTO each render (XMP + IPTC), reads it back, and links nothing that fails
validate_jpeg('stock'), the exposure floor or the label limits; it also writes the Adobe CSV
(`adobe_stock_metadata.csv`, Filename/Title/Keywords/Category/Releases). Only casual
observations are refused; genus/family IDs are labelled at that rank and `grade`/`rank` stay
in the csv. Order of work: `bio stock-calibrate <approved exports>` once, `bio stock-tag-queue
<frames.json> --raw-root F:/Bio --out <csv> --observations <snapshot>`, retouched renders
(`render_stock_candidates.py` in the ops repo, LightCraft develop.auto, no CPU denoise),
`bio stock-publish <csv> ... --gallery-dir <gallery>`. The sharpness/exposure floors are
heuristics, releases (people/property) are NEVER checked here: every row is `release:
unscreened` and the owner's look in the gallery is the gate. A plain raw decode is flat and
un-retouched: `--plain-decode` exists but is off, and the 2048 px iNat renders can never be
Stock files.

**Upload-set flags (2026-10-20).** `bio flag-upload-set <set folder> [--notes json]` appends
`flags`/`flag_notes` columns to the set's `observations.csv` (CRLF kept, re-run replaces them):
`black_frame` is measured (median luma < 2); anything else (not_wildlife, people_visible,
pet, sign_or_text, label_visible, ...) comes from the manual notes file because only a look
at the picture tells. Nothing is removed or reordered -- the owner never culls.

**Remote download (owner works over the LAN, 2026-10-20).** `bio export-zip --kind
inat|stock SRC...` validates every JPEG against the standard first and writes ONE flat
stored zip + `.sha256` into `D:/bio-cache/downloads`; nothing is written when any file
violates it. **The owner does NOT want anything on the tailnet (2026-10-20).** Fetch zips
with `scp user@host:D:/bio-cache/downloads/<name>.zip .` (key auth over the existing sshd).
A `bio serve-downloads` HTTP server (tailnet-identity auth) existed briefly and was REMOVED:
its `Tailscale-User-Login` trust is only safe when `tailscale serve` is the ONLY thing that can
reach the port. It was NOT: the front-door Caddy site `photos.lab.floppydicks.net:4443`
(LAN + tailnet) proxied the same port 8780 and passed a forged header through, so any LAN
client got the zips (reproduced, then closed). Before trusting a header from a loopback
upstream, enumerate every proxy that can forward to that port (`Caddyfile`, `tailscale serve
status`); better, do not hand-roll an HTTP file server at all (the gallery is PiGallery2).
Never reuse `render._check_export_bounds` for validation: it deletes the file it
checks. Persistent services and scheduled tasks need the owner's explicit grant (the
permission classifier blocks them); write the script and hand it over.

**Adobe Stock gallery (2026-10-20).** Framework, not hand-rolled: PiGallery2 3.5.2 (MIT;
native Windows install is unsupported by the maintainer, needs Node 22, so it runs from
`D:/apps/node22`; app in `D:/apps/pigallery2`, thumbnails in `D:/apps/pigallery2-data/tmp`).
It serves `F:/Bio-export/adobe-stock/gallery`, a folder of HARDLINKS (no copies) to the 1,900
shortlist-A files (`F:/bio-automation-data/build_gallery.py`): research grade, species level,
JPEG quality >= q93, >= 24 MP, not tentative. Originals download unchanged (byte-identical,
verified). Listens on 127.0.0.1:8781 ONLY (the tailscale serve entry was removed at the
owner's request); app login `owner` (password in `F:/services/bio-gallery/password.txt`; the
default admin/admin was deleted by `gallery_secure.py`). **Persistent:** scheduled task
`Bio-Gallery` (S4U, startup + logon, restart on failure) runs the supervisor
`F:/services/bio-gallery/run-gallery.ps1`, which waits for the photo folder, kills an orphaned
PiGallery2 `node.exe` before each start (Stop-ScheduledTask ends the supervisor but NOT its
child) and logs to `D:/apps/pigallery2-data/server.log` via `cmd` redirection. Runs with
`--Server-trustProxy=loopback`, so failed logins log the real client address and the session
cookie is `Secure`. Reboot survival not tested. **LAN access (applied
2026-10-20 with the owner's yes):** https://photos.lab.floppydicks.net:4443/ . The name
resolves to the LAN IP 172.16.100.11; the front-door Caddy block of that name now binds ONLY
172.16.100.11, answers 403 unless the client is in 172.16.100.0/24 or loopback, and proxies
to 127.0.0.1:8781 (wildcard Let's Encrypt cert, HSTS, PiGallery2 login behind it). Not on
the tailnet. Verified from this host and from mrpink (172.16.100.20): TLS valid, front page
200, anonymous API 401, original byte-identical, thumbnails cacheable. The 403 branch was
confirmed in Caddy's compiled config only (no client outside the /24 could be sourced).
Rollback: copy `F:/services/tls/caddy/Caddyfile.bak-photos-lan-20261020-091908` over
`Caddyfile`, then Stop+Start scheduled task `Lab-TLS-FrontDoor` (blips the loci/litellm/ollama
names ~5 s). Caddyfile has MIXED newlines (LF + one trailing CRLF): never assume one style.

**Bluesky posts (owner, 2026-10-21; `social.py`, folded in from github.com/rjmendez/inat_skeet).** Daily posts of the
owner's photos from the CURRENT month of every year in the gallery (`<year>/<year>-<MM>/`), one iNat observation per post,
gallery renders only (never iNat's 2048 px copies). Alt text comes from the LOCAL vision model (env `BIO_VISION_MODEL`, else
the pool's `vision` role; never a literal model name), given the embedded Stock label and the live observation; GPU only.
**Every post is approved by the owner** (`bio social-approve`); nothing auto-approves, and a draft edited after approval needs a
new approval. `bio social-post` is dry-run unless `--live`, posts at most `BIO_SOCIAL_POSTS_PER_DAY` a day, and a send that
raised leaves the draft `posting`, which blocks all posting until the owner checks the profile and runs `bio social-resolve`
(never resend on a guess). Never print a private-geoprivacy place. Alt text follows the owner's own style
(`bio social-style` -> `settings.SOCIAL_STYLE_PROFILE`, learned from alt texts they wrote on Bluesky, inat_skeet bot posts
excluded; deterministic statistics + rules + nearest examples, nothing model-written; drafting refuses to run without it;
mirrored in Loci investigation `bsky-alt-text-style`). Daily task `Bio-Social-Daily` runs
`F:/services/bio-social/run-social.ps1` (index -> draft -> post). Images go up at the official app's post limits (4000 px,
2,000,000 B; `BSKY_IMAGE_MAX_*`); the 2 MB cap is the lexicon's, so no other upload path can carry more. A print link
("Prints: <url>", e.g. the Darkroom product page) is added from the owner's `settings.SOCIAL_PRINTS_FILE` csv (`obs_id,url`);
Darkroom has no API, so only the owner fills it; hashtags are dropped before the print link when the text is over 300.

**Removing the old inat_skeet posts (owner, 2026-10-22: "all without replies", "dry-run list, then one go").**
`bio social-cleanup-plan` lists, from live public data, every post in inat_skeet's template that has no replies (posts with
replies are kept so no conversation is orphaned) and saves the plan with an id; `bio social-cleanup-run <plan> --confirm <id>`
deletes exactly that list through the separate delete-only "cleanup" profile (its own client, key, session and sign-in;
scope `repo:app.bsky.feed.post?action=delete`), resumable via a ledger next to the plan. The full account export is archived in
`settings.BSKY_ARCHIVE_DIR` first. Never delete anything outside an owner-confirmed plan.

**Sign-in is atproto OAuth through Bluesky's own client library (owner, 2026-10-22): use a mature, maintained
implementation and never roll your own OAuth, DPoP, JWT or other crypto step.** Posting goes through the `bsky-oauth` library
(private repo `rjmendez/bsky-oauth`, extra `[social]`, then `bsky-oauth install`), which is glue around
`@atproto/oauth-client-node`, `@atproto/jwk-jose` and `@atproto/api` at pinned versions. bio-automation is its "poster" profile
(`social.poster_profile()`): a confidential client (a public one is capped at 2-week sessions) whose metadata and PUBLIC key
are published by repo `rjmendez/bio-social-oauth` (GitHub Pages, `settings.BSKY_CLIENT_ID`); the private key and the session
stay in `settings.BSKY_OAUTH_DIR`. Scope: create posts and upload images only; other bots get their own profile and scope. The
owner signs in once with `bio social-login`; the daily run keeps the session alive by refreshing it.

**No web UI work (owner, 2026-10-20).** The review console / inat_console web UIs
are shelved: do not build, extend, audit or fix them. The focus is bulk photo
processing from the command line (render, tag, validate, build upload sets).
Rebuilds of derived data and restarts of local services are pre-authorized only
with a backup, a verification step and a rollback noted; posting, DNS/Caddy,
deleting originals and pushing stay ask-first.

## Test-writing rules

These apply to every test added or edited in this repo, no exceptions:

1. **Assert the exact value the claim names.** Not "key present", not a range,
   not `>= 0`. If the right answer is known, pin it.
2. **Never stub, re-implement or source-grep the unit under test.** Stub only its
   dependencies.
3. **Every negative test has a positive twin on the same fixture.**
4. **A stub that must not be called records its calls; assert the list is empty.**
5. **Error, degraded or fail-open output is not a pass.** Fake the dependency so
   the success branch actually runs and its values get asserted.
6. **`pytest.raises(SpecificError, match=...)`.** A bare/broad exception class
   without a real `match=` is rejected.
7. **The fixture must be able to tell right from wrong**: distinct values,
   boundary cases, winners that aren't first-by-accident.
8. **Never pin a known bug as expected.** Use `xfail(strict=True, reason=...)`
   instead.
9. **No state leaks.** `monkeypatch` for globals/env, `tmp_path` for files.
10. **No off-by-default tests.** A test gated behind an opt-in env var or marker
    that nothing turns on in normal use never runs, which makes it no test. If a
    real external dependency (hardware, a license-gated binary, a slow model) is
    genuinely required, still prefer making it run by default against a small
    committed fixture over skipping it — see `tests/test_preclean_live.py` for the
    pattern: a real raw file + real model, cropped down so it still runs in every
    `pytest` invocation at low enough cost to not be worth skipping.
11. **Prove it before claiming coverage.** Apply the obvious mutation to the code
    (delete the guard, return the constant, invert the check), confirm the test
    goes red, revert. State which mutation was run.
12. **No test without an assertion; no `assert True`; no swallowed `AssertionError`.**

### Enforcement

`scripts/check_test_honesty.py` mechanically rejects rules 1 (broad-raises),
8 (xfail-not-strict), 10 (env-skipif) and 12 (no-assertions/assert-true/or-true/
swallowed-assert):

    python scripts/check_test_honesty.py              # whole repo
    python scripts/check_test_honesty.py path/to/test.py

Its allowlist (`scripts/test_honesty_allowlist.toml`) starts empty and should stay
that way — fix the test rather than adding an entry.

## Running tests

`pytest` from the repo root. `tests/test_preclean_live.py` is real and unmocked
(real rawpy decode of `tests/fixtures/sample_raw.NEF`, real OpenVINO model
inference) and runs by default — it adds roughly 40-60s to the suite. That cost is
deliberate (see rule 10 above); do not add a skip condition to make it faster.
