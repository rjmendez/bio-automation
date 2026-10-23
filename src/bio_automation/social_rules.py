"""The rules behind the Bluesky posts: every one optional, every decision audited (owner, 2026-10-22).

A rule is a named switch (and, for the daily cap, a number). Where a rule gets its state, last wins:
  1. its default below;  2. the older settings (BIO_SOCIAL_POSTS_PER_DAY, BIO_SOCIAL_AUTO_APPROVE, BIO_SOCIAL_THEMES);
  3. the owner's TOML file (settings.SOCIAL_RULES_FILE);  4. the environment, BIO_SOCIAL_RULE_<NAME> = on | off | <number>.
`bio social-rules` shows each rule with its state and where the state came from; `--write-defaults` writes the file to edit.

Every decision that applies or skips a rule is appended to the audit log (settings.SOCIAL_AUDIT_LOG, JSON lines): which rules were on, which
draft was chosen and why, what was skipped and why, what was posted. A post is not sent when its decision cannot be written."""
from __future__ import annotations

import datetime as dt
import json
import os
import tomllib
import uuid
from pathlib import Path
from typing import Mapping, NamedTuple

from . import secrets_guard, settings


class RulesError(Exception):
    pass


class Rule(NamedTuple):
    name: str
    default: bool
    value: int | None                 # a number for the rules that have one (daily_cap), else None
    description: str


RULES: tuple[Rule, ...] = (
    Rule("daily_cap", True, 3, "At most <value> scheduled posts a day (the 08:00, 12:00 and 20:00 runs)."),
    Rule("manual_exempt_from_cap", True, None, "Manual posts (--manual, --draft) are neither counted in nor stopped by daily_cap."),
    Rule("auto_approve", True, None, "A scheduled run approves the oldest drafted draft itself when none is approved (no per-draft review)."),
    Rule("month_candidates", False, None, "Draft only observations photographed in the current month of any year."),
    Rule("month_posting", False, None, "Post only drafts of the current month; an earlier month's unposted drafts expire."),
    Rule("themes", True, None, "Weekday theme posts: up to 4 photos of one kind of organism in one post (Mushroom Monday ...)."),
    Rule("theme_day_only", True, None, "A themed draft posts only on its own day, before the single posts; off: it joins the ordinary queue."),
    Rule("theme_species_first", True, None, "A theme group takes one observation per species, species-level IDs before coarse ones."),
    Rule("lower_quality_photos", True, 1000, "Observations without a gallery photo are eligible too, no print needed, with their best iNat photo (its original size from iNat's open data, fetched when the draft is made) when it is at least <value> px on the long side; drafts with a smaller photo are withdrawn."),
    Rule("prints_first", True, None, "Observations with a Darkroom print are drafted first."),
    Rule("year_rotation", True, None, "Candidates take turns by year, newest first; off: one shuffled pool."),
    Rule("skip_casual", True, None, "Casual iNat observations are never drafted."),
    Rule("strip_review_claim", True, None, "Remove the old 'Alt text: local AI model, reviewed.' line from a draft when it is approved."),
    Rule("alt_text_privacy", True, None, "Alt text never repeats a private place: a draft whose alt text names a place the post does not show is not drafted."),
    Rule("live_recheck", True, None, "Just before a post is sent its observation is read from iNat again; a deleted, casual or changed one is withdrawn."),
    Rule("image_floor", True, None, "A draft whose photos fall below the quality floor (size, sharpness) is not drafted or posted."),
    Rule("pause_switch", True, None, "A PAUSED file in the social folder stops every scheduled post until the file is removed."),
    Rule("circuit_breaker", True, 3, "After <value> failed runs in a row posting stops until `bio social-breaker-reset` (or the next successful dry check)."),
    Rule("verify_posts", True, None, "After a post is sent it is read back from Bluesky; a post that is not there is reported."),
    Rule("alerts", True, None, "A failed run, a blocked post or a tripped breaker is written to the alerts log (and passed to the alert command, if set)."),
    Rule("ledger_integrity", True, None, "The ledger is checked before a post and backed up daily; a corrupt line stops posting."),
    Rule("variety", True, 3, "No two of the last <value> posts share a kind of organism, place or day-of-week theme when another draft is eligible."),
    Rule("seasonal_boost", False, None, "Observations from the current season of any year are drafted first."),
    Rule("photo_cache_cap", True, 10, "The downloaded iNat photo cache is kept under <value> GB: the oldest photos are deleted first."),
)
BY_NAME = {r.name: r for r in RULES}


class RuleState(NamedTuple):
    enabled: bool
    value: int | None
    source: str                       # default | settings | file | env


class Rules:
    def __init__(self, states: Mapping[str, RuleState]):
        self._s = dict(states)

    def on(self, name: str) -> bool:
        return self._s[name].enabled

    def value(self, name: str) -> int | None:
        return self._s[name].value

    def state(self, name: str) -> RuleState:
        return self._s[name]

    def snapshot(self) -> dict[str, dict]:
        """{rule: {"on": bool, "value": number or None}}: what an audit record carries."""
        return {n: {"on": s.enabled, **({"value": s.value} if s.value is not None else {})} for n, s in self._s.items()}


def _flag(text: str, where: str, has_value: bool = False):
    t = str(text).strip().lower()
    if has_value and t.isdigit() and int(t) >= 1:                      # for a rule with a number, 1 is the number 1, not "on"
        return True, int(t)
    if t in ("on", "1", "true", "yes"):
        return True, None
    if t in ("off", "0", "false", "no", ""):
        return False, None
    if t.lstrip("-").isdigit():
        return True, int(t)
    raise RulesError(f"{where}: expected on, off or a number, got {text!r}")


def load_rules(path: Path | None = None, env: Mapping[str, str] | None = None) -> Rules:
    """The rules as the file and the environment say; RulesError names the rule and where the bad value is."""
    path = Path(path or settings.SOCIAL_RULES_FILE)
    env = os.environ if env is None else env
    st = {r.name: RuleState(r.default, r.value, "default") for r in RULES}
    for name, enabled, value in (("daily_cap", True, settings.SOCIAL_POSTS_PER_DAY), ("auto_approve", settings.SOCIAL_AUTO_APPROVE, None),
                                 ("themes", settings.SOCIAL_THEMES, None)):
        if (enabled, value) != (BY_NAME[name].default, BY_NAME[name].value):
            st[name] = RuleState(enabled, value, "settings")
    if path.is_file():
        try:
            doc = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except tomllib.TOMLDecodeError as e:
            raise RulesError(f"{path}: {e}") from e
        for name, body in (doc.get("rules") or {}).items():
            if name not in BY_NAME:
                raise RulesError(f"{path}: unknown rule {name!r} (rules: {', '.join(BY_NAME)})")
            enabled = body.get("enabled", st[name].enabled)
            value = body.get("value", st[name].value)
            if not isinstance(enabled, bool):
                raise RulesError(f"{path}: rule {name!r}: enabled must be true or false")
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise RulesError(f"{path}: rule {name!r}: value must be a whole number")
            if value is not None and BY_NAME[name].value is None:
                raise RulesError(f"{path}: rule {name!r} has no value")
            st[name] = RuleState(enabled, value, "file")
    for name in BY_NAME:
        raw = env.get("BIO_SOCIAL_RULE_" + name.upper())
        if raw is not None:
            enabled, value = _flag(raw, f"BIO_SOCIAL_RULE_{name.upper()}", BY_NAME[name].value is not None)
            if value is not None and BY_NAME[name].value is None:
                raise RulesError(f"BIO_SOCIAL_RULE_{name.upper()}: rule {name!r} has no value, use on or off")
            st[name] = RuleState(enabled, value if value is not None else st[name].value, "env")
    return Rules(st)


def rules_toml(rules: Rules | None = None) -> str:
    """The rules as the TOML file load_rules() reads back: the starting point for the owner's own file."""
    rules = rules or load_rules(Path(os.devnull), env={})
    out = ["# Bluesky posting rules (bio-automation social). Every rule is optional: set enabled = false to switch it off.",
           "# State order, last wins: default, the old BIO_SOCIAL_* settings, this file, BIO_SOCIAL_RULE_<NAME> in the environment.", ""]
    for r in RULES:
        s = rules.state(r.name)
        out += [f"# {r.description}", f"[rules.{r.name}]", f"enabled = {str(s.enabled).lower()}"]
        if r.value is not None:
            out.append(f"value = {s.value}")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------- the audit log

class Audit:
    """Append-only JSON lines: {"at", "run", "event", ...}. One run id per object, so a run's records can be read together."""

    def __init__(self, path: Path | None = None, *, run: str | None = None, now=None):
        self.path = Path(path or settings.SOCIAL_AUDIT_LOG)
        self.run = run or uuid.uuid4().hex[:12]
        self._now = now or (lambda: dt.datetime.now(settings.LOCAL_TZ))

    def log(self, event: str, **fields) -> dict:
        rec = {"at": self._now().isoformat(timespec="seconds"), "run": self.run, "event": event, **secrets_guard.redact(fields)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        return rec

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
