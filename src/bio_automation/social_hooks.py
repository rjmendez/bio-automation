"""The hook framework behind the Bluesky posting workflow: small, named, rule-gated checks that plug into fixed stages of the flow.

A hook is fn(ctx: dict) -> Verdict | None (None is a pass). It runs only while its rule is ON (rule=None: always), lower priority first.
Every non-pass verdict is audited (event "hook"); an exception inside a hook is audited (event "hook_error") and then handled per stage:
a pass for draft, order, post_send and maintenance (a broken check must not stop the drafting), a BLOCK for pre_send (a broken check must
never let a post through: fail closed). A CRITICAL hook (register(..., critical=True): the privacy gates) fails closed in every stage that can
refuse: a reject for draft, a block for pre_send (order, post_send and maintenance cannot refuse, so those pass); the audit says outcome=reject|block|pass.

Stages and their ctx:
  draft        {obs, obs_id, image (Path), alt, label, month, rules}        reject = skip this observation
  order        {candidates (list of obs ids, mutate IN PLACE), kind ("single" | "theme:<key>"), taxonomy, ledger, today, rules}
  pre_send     {draft_id, draft, draft_dir, ledger, now, live, manual, fetch, rules}   reject/withdraw = withdraw the draft; block = send nothing
  post_send    {draft_id, draft, uri, cid, request, ledger, now, manual, rules}        results are audited only
  maintenance  {drafts_dir, ledger, rules, today}                            after each social-draft run
draft and pre_send stop at the first non-pass verdict; the other stages run every hook.

Extension modules (EXTENSIONS) are imported lazily, each on its own: one that fails to import or to register is recorded in failed() and audited
(hook_error, stage "load"), and the other commands go on; require_loaded() raises ExtensionsFailed for the commands that need the gates (GATED_COMMANDS); each may define register_hooks(), add_parsers(sub) and COMMANDS = {command: fn(args) -> int}
(and ERRORS, a tuple of exception classes the CLI prints as one line)."""
from __future__ import annotations

import importlib
from typing import Callable, NamedTuple

STAGES = ("draft", "order", "pre_send", "post_send", "maintenance")
ACTIONS = ("pass", "reject", "withdraw", "block")
STOP_AT_FIRST = ("draft", "pre_send")
FAIL_CLOSED = ("pre_send",)
GATED_COMMANDS = ("social-post", "social-draft", "social-prep")      # the commands that post or draft: they must not run without every extension's gates
EXTENSIONS = ("social_gates", "social_controls", "social_verify", "social_alerts", "social_prep", "social_selfcheck", "social_state",
              "social_taste", "social_engagement")
PACKAGE = __package__ or "bio_automation"


class ExtensionsFailed(Exception):
    """An extension module did not load, so its gates would be missing."""


class Verdict(NamedTuple):
    action: str                       # pass | reject | withdraw | block
    reason: str = ""


class Hook(NamedTuple):
    name: str
    stage: str
    rule: str | None
    fn: Callable[[dict], "Verdict | None"]
    priority: int
    critical: bool = False


_hooks: list[Hook] = []
_loaded: set[str] = set()
_modules: dict[str, object] = {}
_failed: dict[str, str] = {}


def register(name: str, stage: str, fn: Callable[[dict], "Verdict | None"], *, rule: str | None = None, priority: int = 50,
             critical: bool = False) -> Hook:
    """Add a hook (a second registration of the same name and stage replaces the first, so registering twice is harmless). A critical hook
    fails closed in every stage when it raises (see the module docstring)."""
    if stage not in STAGES:
        raise ValueError(f"unknown hook stage {stage!r} (stages: {', '.join(STAGES)})")
    hook = Hook(name, stage, rule, fn, priority, critical)
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
            if stage in FAIL_CLOSED:
                verdict = Verdict("block", f"hook {hook.name} failed ({error}); nothing is sent")
            elif hook.critical and stage == "draft":
                verdict = Verdict("reject", f"critical hook {hook.name} failed ({error}); the draft is rejected")
            else:
                verdict = Verdict("pass", f"hook error: {error}")
            _log(audit, "hook_error", name=hook.name, stage=stage, error=error, outcome=verdict.action, **({"critical": True} if hook.critical else {}), **_who(ctx))
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


def failed() -> dict[str, str]:
    """{extension module: error} for the extensions that did not load (a copy)."""
    return dict(_failed)


def _record_failure(name: str, e: BaseException) -> None:
    error = f"{type(e).__name__}: {e}"[:300]
    if _failed.get(name) == error:
        return                                                  # the same failure seen again (every extensions() call retries): audited once
    _failed[name] = error
    try:
        from .social_rules import Audit

        Audit().log("hook_error", name=name, stage="load", error=error, outcome="skipped")
    except Exception:  # noqa: BLE001 -- an unwritable audit log never turns a skipped extension into a crash
        pass


def extensions() -> list:
    """The extension modules that load. One that is missing (every name in EXTENSIONS must exist in a shipped tree) or fails to import is skipped
    and recorded in failed() (and audited): a broken extension must not take every other command down, but it is never silent."""
    out = []
    for name in EXTENSIONS:
        full = _qualified(name)
        if full not in _modules:
            try:
                _modules[full] = importlib.import_module(full)
            except Exception as e:                       # a module that is not there (deleted, not shipped) is a failure too: its gates are missing  # noqa: BLE001 -- any import-time error of one extension
                _record_failure(name, e)
                continue
            _failed.pop(name, None)
        out.append(_modules[full])
    return out


def load_all() -> list:
    """Import every extension and call its register_hooks() once (a second call does nothing more). Returns the modules that loaded; the ones
    that did not are in failed() (see require_loaded())."""
    mods = []
    for mod in extensions():
        if mod.__name__ not in _loaded:
            name = mod.__name__.rsplit(".", 1)[-1]
            try:
                if hasattr(mod, "register_hooks"):
                    mod.register_hooks()
            except Exception as e:  # noqa: BLE001
                _record_failure(name, e)
                continue
            _loaded.add(mod.__name__)
            _failed.pop(name, None)
        mods.append(mod)
    return mods


def require_loaded() -> list:
    """load_all(), but raise ExtensionsFailed when any extension did not load: for the code that posts or drafts, which needs every gate."""
    mods = load_all()
    if _failed:
        raise ExtensionsFailed("these social modules did not load, so their gates are missing and nothing is posted or drafted: "
                               + "; ".join(f"{n} ({e})" for n, e in sorted(_failed.items())))
    return mods


def reset() -> None:
    """Forget every hook and every loaded extension (tests)."""
    _hooks.clear()
    _loaded.clear()
    _modules.clear()
    _failed.clear()
