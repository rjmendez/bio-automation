"""The GitHub workflow: parsed, not grepped. Pins, credential handling and the degraded (no secret) path are properties of the parsed YAML."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
SECRET = "BSKY_OAUTH_READ_TOKEN"
PRIVATE_FILE = "tests/test_social_cleanup.py"
PRIVATE_CLASSES = ("TestBskyBoundary", "TestLocalFailuresBeforeTheWrite")


def load():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps(job):
    return load()["jobs"][job]["steps"]


def uses_steps(job):
    return [s for s in steps(job) if "uses" in s]


def unit_test_command(job, label):
    (s,) = [s for s in steps(job) if s.get("name", "").startswith(label)]
    return s["run"].split()


class TestWorkflowStructure:
    def test_jobs_and_the_secretless_public_job(self):
        wf = load()
        assert list(wf["jobs"]) == ["gate", "tests", "tests-private", "summary"]
        assert wf["jobs"]["tests"]["name"] == "tests"          # a branch-protection rule that requires the check `tests` keeps matching
        assert wf["permissions"] == {"contents": "read"}
        assert wf["jobs"]["tests"]["needs"] == "gate" and "if" not in wf["jobs"]["tests"]       # the public job always runs
        assert wf["jobs"]["tests-private"]["if"] == "${{ needs.gate.outputs.have_token == 'true' }}"

    def test_every_action_is_pinned_to_a_40_hex_sha_and_does_not_persist_credentials_when_it_checks_out(self):
        for job in ("tests", "tests-private"):
            for s in uses_steps(job):
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", s["uses"]), s["uses"]
                if s["uses"].startswith("actions/checkout@"):
                    assert s["with"]["persist-credentials"] is False
        assert [s["uses"].split("@")[0] for s in uses_steps("tests")] == ["actions/checkout", "actions/setup-python"]
        assert [s["uses"].split("@")[0] for s in uses_steps("tests-private")] == ["actions/checkout", "actions/setup-python", "actions/checkout"]

    def test_the_token_is_only_the_token_input_of_the_private_checkout_and_the_gate_compares_it_to_empty(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        uses = re.findall(r"\$\{\{[^}]*" + SECRET + r"[^}]*\}\}", text)
        assert sorted(uses) == sorted(["${{ secrets.%s != '' }}" % SECRET, "${{ secrets.%s }}" % SECRET])
        wf = load()
        (priv,) = [s for s in steps("tests-private") if s.get("with", {}).get("repository") == "rjmendez/bsky-oauth"]
        assert priv["with"]["token"] == "${{ secrets.%s }}" % SECRET
        assert "env" not in priv
        assert "secrets." not in str(steps("tests"))                  # the public job never reads the secrets context
        (probe,) = wf["jobs"]["gate"]["steps"]
        assert probe["env"] == {"HAVE_TOKEN": "${{ secrets.%s != '' }}" % SECRET}
        for job in ("tests", "tests-private"):
            for s in steps(job):
                if s.get("with", {}).get("repository") != "rjmendez/bsky-oauth":
                    assert SECRET not in str(s.get("env", {})) and SECRET not in str(s.get("with", {}))

    def test_the_private_checkout_is_pinned_to_a_commit(self):
        (priv,) = [s for s in steps("tests-private") if s.get("with", {}).get("repository") == "rjmendez/bsky-oauth"]
        assert re.fullmatch(r"[0-9a-f]{40}", priv["with"]["ref"])


class TestDegradedRun:
    def test_the_public_job_installs_no_bsky_oauth(self):
        installs = [s["run"] for s in steps("tests") if "pip install -e ." in s.get("run", "")]
        assert len(installs) == 1 and "bsky-oauth" not in installs[0]
        assert "bsky-oauth install" not in str(steps("tests"))

    def test_the_public_job_leaves_out_exactly_the_private_tests_and_the_private_job_leaves_out_nothing_more(self):
        public = unit_test_command("tests", "Unit tests")
        private = unit_test_command("tests-private", "Unit tests")
        extra = [a for a in public if a not in private]
        assert extra == ["--ignore=" + PRIVATE_FILE] + [f"--deselect=tests/test_social.py::{c}" for c in PRIVATE_CLASSES]
        assert [a for a in private if a not in public] == []

    def test_the_left_out_targets_exist(self):
        assert (REPO / PRIVATE_FILE).is_file()
        tree = ast.parse((REPO / "tests" / "test_social.py").read_text(encoding="utf-8"))
        classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        assert set(PRIVATE_CLASSES) <= classes

    def test_every_left_out_target_really_needs_bsky_oauth_and_no_other_test_file_imports_it_at_module_level(self):
        """Positive and negative twin: the left-out file and classes import bsky_oauth; no other test module does at the top level."""
        def module_level_imports(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and "bsky_oauth" in ast.dump(n)]
        assert len(module_level_imports(REPO / PRIVATE_FILE)) == 1
        others = [p.name for p in sorted((REPO / "tests").glob("test_*.py")) if p.name != "test_social_cleanup.py" and module_level_imports(p)]
        assert others == []
        tree = ast.parse((REPO / "tests" / "test_social.py").read_text(encoding="utf-8"))
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in PRIVATE_CLASSES):
            assert "bsky_oauth" in ast.unparse(cls) or "bsky_post" in ast.unparse(cls)

    def test_the_gate_and_the_public_job_say_when_the_private_part_did_not_run(self):
        (probe,) = load()["jobs"]["gate"]["steps"]
        assert "PARTIAL RUN: the private-library tests did NOT run" in probe["run"]
        assert "::warning title=private tests NOT run::" in probe["run"]
        (note,) = [s for s in steps("tests") if s.get("name") == "What this run left out"]
        assert note["if"] == "${{ always() }}"
        assert "::warning title=partial run::" in note["run"]
        assert "tests/test_social_cleanup.py" in note["run"] and "TestBskyBoundary" in note["run"]


class TestSummaryJob:
    def test_it_reads_both_results_runs_always_and_fails_on_a_failed_or_cancelled_job(self):
        job = load()["jobs"]["summary"]
        assert (job["needs"], job["if"], job["name"]) == (["gate", "tests", "tests-private"], "${{ always() }}", "tests-summary")
        (step,) = job["steps"]
        assert step["env"] == {"PUBLIC": "${{ needs.tests.result }}", "PRIVATE": "${{ needs.tests-private.result }}"}
        run = step["run"]
        assert 'if [ "$PUBLIC" != "success" ]; then' in run
        assert 'if [ "$PRIVATE" = "failure" ] || [ "$PRIVATE" = "cancelled" ]; then' in run

    def test_a_skipped_private_job_ends_in_an_error_annotation_not_a_silent_pass(self):
        (step,) = load()["jobs"]["summary"]["steps"]
        assert "::error title=PRIVATE TESTS DID NOT RUN::" in step["run"]
        assert 'if [ "$PRIVATE" != "success" ]; then' in step["run"]

    def test_the_summary_script_behaves_for_each_result(self, tmp_path):
        import os, shutil, subprocess
        bash = shutil.which("bash")
        (step,) = load()["jobs"]["summary"]["steps"]
        outcomes = {}
        for public, private in (("success", "success"), ("success", "skipped"), ("failure", "skipped"), ("success", "failure")):
            r = subprocess.run([bash, "-c", step["run"]], env={**os.environ, "PUBLIC": public, "PRIVATE": private, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")},
                               capture_output=True, text=True)
            outcomes[(public, private)] = (r.returncode, "PRIVATE TESTS DID NOT RUN" in r.stdout)
        assert outcomes == {("success", "success"): (0, False), ("success", "skipped"): (0, True), ("failure", "skipped"): (1, False), ("success", "failure"): (1, False)}
