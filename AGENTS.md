# AGENTS.md

Instructions for coding agents working in this repository.

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
