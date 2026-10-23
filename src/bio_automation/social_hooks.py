"""The hook framework behind the Bluesky posting workflow: small, named, rule-gated checks that plug into fixed stages of the flow.

A hook is fn(ctx: dict) -> Verdict | None (None is a pass). It runs only while its rule is ON (rule=None: always), lower priority first.
Every non-pass verdict is audited (event "hook"); an exception inside a hook is audited (event "hook_error") and then handled per stage:
a pass for draft, order, post_send and maintenance (a broken check must not stop the drafting), a BLOCK for pre_send (a broken check must
never let a post through: fail closed).

Stages and their ctx:
  draft        {obs, obs_id, image (Path), alt, label, month, rules}        reject = skip this observation
  order        {candidates (list of obs ids, mutate IN PLACE), kind ("single" | "theme:<key>"), taxonomy, ledger, today, rules}
  pre_send     {draft_id, draft, draft_dir, ledger, now, live, manual, fetch, rules}   reject/withdraw = withdraw the draft; block = send nothing
  post_send    {draft_id, draft, uri, cid, request, ledger, now, manual, rules}        results are audited only
  maintenance  {drafts_dir, ledger, rules, today}                            after each social-draft run
draft and pre_send stop at the first non-pass verdict; the other stages run every hook.

Extension modules (EXTENSIONS) are imported lazily; each may define register_hooks(), add_parsers(sub) and COMMANDS = {command: fn(args) -> int}
(and ERRORS, a tuple of exception classes the CLI prints as one line)."""
from __future__ import annotations

import importlib
from typing import Callable, NamedTuple

STAGES = ("draft", "order", "pre_send", "post_send", "maintenance")
ACTIONS = ("pass", "reject", "withdraw", "block")
STOP_AT_FIRST = ("draft", "pre_send")
FAIL_CLOSED = ("pre_send",)
EXTENSIONS = ("social_gates", "social_controls", "social_verify", "social_alerts", "social_prep", "social_selfcheck", "social_state",
              "social_taste", "social_engagement")
PACKAGE = __package__ or "bio_automation"


class Verdict(NamedTuple):
    action: str                       # pass | reject | withdraw | block
    reason: str = ""


class Hook(NamedTuple):
    name: str
    stage: str
    rule: str | None
    fn: Callable[[dict], "Verdict | None"]
    priority: int


_hooks: list[Hook] = []
_loaded: set[str] = set()
_modules: dict[str, object] = {}


def register(name: str, stage: str, fn: Callable[[dict], "Verdict | None"], *, rule: str | None = None, priority: int = 50) -> Hook:
    """Add a hook (a second registration of the same name and stage replaces the first, so registering twice is harmless)."""
    if stage not in STAGES:
        raise ValueError(f"unknown hook stage {stage!r} (stages: {', '.join(STAGES)})")
    hook = Hook(name, stage, rule, fn, priority)
    _hooks[:] = [h for h in _hooks if (h.name, h.stage) != (name, stage)] + [hook]
    return hook


def hooks(stage: str) -> list[Hook]:
    """The stage's hooks in the order they run: lower priority first, then in registration order."""
    return sorted((h for h in _hooks if h.stage == stage), key=lambda h: h.priority)


def _who(ctx: dict) -> dict:
    out = {}
    if ctx.get("draft_id") is not None:
        out["draft_id"] = ctx["draft_id"]
    if ctx.get("obs_id") is not None:
        out["obs_id"] = ctx["obs_id"]
    return out


def _log(audit, event: str, **fields) -> None:
    try:
        audit.log(event, **fields)
    except OSError:
        pass                                    # an unwritable audit log never changes a verdict; the callers that need it (the send) check it themselves


def _verdict(raw) -> Verdict:
    if raw is None:
        return Verdict("pass")
    v = raw if isinstance(raw, Verdict) else Verdict(*raw)
    if v.action not in ACTIONS:
        raise ValueError(f"unknown verdict action {v.action!r} (actions: {', '.join(ACTIONS)})")
    return v


def run(stage: str, ctx: dict, *, rules, audit) -> list[tuple[Hook, Verdict]]:
    """Run the stage's hooks whose rule is on; return (hook, verdict) for each that ran (draft and pre_send: up to and including the first
    non-pass one). Non-pass verdicts and errors go to the audit log."""
    if stage not in STAGES:
        raise ValueError(f"unknown hook stage {stage!r} (stages: {', '.join(STAGES)})")
    results: list[tuple[Hook, Verdict]] = []
    for hook in hooks(stage):
        try:
            if hook.rule is not None and not rules.on(hook.rule):
                continue
            verdict = _verdict(hook.fn(ctx))
        except Exception as e:  # noqa: BLE001 -- a hook is untrusted code to the flow it plugs into
            error = f"{type(e).__name__}: {e}"[:300]
            closed = stage in FAIL_CLOSED
            _log(audit, "hook_error", name=hook.name, stage=stage, error=error, outcome="block" if closed else "pass", **_who(ctx))
            verdict = Verdict("block", f"hook {hook.name} failed ({error}); nothing is sent") if closed else Verdict("pass", f"hook error: {error}")
        results.append((hook, verdict))
        if verdict.action != "pass":
            _log(audit, "hook", name=hook.name, stage=stage, action=verdict.action, reason=verdict.reason, **_who(ctx))
            if stage in STOP_AT_FIRST:
                break
    return results


def first_non_pass(results: list[tuple[Hook, Verdict]]) -> tuple[Hook, Verdict] | None:
    for hook, verdict in results:
        if verdict.action != "pass":
            return hook, verdict
    return None


# --------------------------------------------------------------------------------------------------- extension modules

def _qualified(name: str) -> str:
    return name if "." in name else f"{PACKAGE}.{name}"


def extensions() -> list:
    """The extension modules that exist. A module that is not there yet is skipped; an import error inside one that is there propagates."""
    out = []
    for name in EXTENSIONS:
        full = _qualified(name)
        if full not in _modules:
            try:
                _modules[full] = importlib.import_module(full)
            except ModuleNotFoundError as e:
                if e.name != full:
                    raise
                continue
        out.append(_modules[full])
    return out


def load_all() -> list:
    """Import every extension and call its register_hooks() once (a second call does nothing more). Returns the modules."""
    mods = extensions()
    for mod in mods:
        if mod.__name__ in _loaded:
            continue
        if hasattr(mod, "register_hooks"):
            mod.register_hooks()
        _loaded.add(mod.__name__)
    return mods


def reset() -> None:
    """Forget every hook and every loaded extension (tests)."""
    _hooks.clear()
    _loaded.clear()
    _modules.clear()
