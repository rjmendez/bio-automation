"""lightcraft_mcp: the guard against the wrong catalog, and the id mapping of an import (with a recording fake engine, no LightCraft process)."""
import pytest

from bio_automation import lightcraft_mcp as lm

LIB = r"C:\Users\x\Pictures\LightCraft Library"


def info(**kw):
    d = {"persistent": True, "path": LIB, "photos": 16620, "load": {"damaged": None}}
    d.update(kw)
    return d


class TestCheckLibraryInfo:
    def test_the_persistent_expected_library_passes(self):
        assert lm.check_library_info(info(), lm.Path(LIB)) is None

    def test_path_comparison_ignores_case_and_separators(self):
        assert lm.check_library_info(info(path=LIB.lower().replace("\\", "/")), lm.Path(LIB)) is None

    def test_a_scratch_catalog_is_refused(self):
        with pytest.raises(lm.LightCraftLibraryError, match=r"is not persistent: edits and imports would be lost"):
            lm.check_library_info(info(persistent=False), lm.Path(LIB))

    def test_another_library_is_refused(self):
        with pytest.raises(lm.LightCraftLibraryError, match=r"engine opened library 'D:\\\\other', expected"):
            lm.check_library_info(info(path=r"D:\other"), lm.Path(LIB))

    def test_a_damaged_load_is_refused(self):
        with pytest.raises(lm.LightCraftLibraryError, match=r"loaded damaged: 'torn snapshot'"):
            lm.check_library_info(info(load={"damaged": "torn snapshot"}), lm.Path(LIB))


class FakeLib:
    def __init__(self, result):
        self.result, self.calls = result, []

    def call(self, tool, args):
        self.calls.append((tool, args))
        return self.result


class TestImportFiles:
    def test_new_and_duplicate_files_all_get_their_ids(self):
        fake = FakeLib({"imported": [101, 103], "duplicates": [{"path": "b.nef", "existing": 55}], "failed": []})
        res = lm.import_files(fake, ["a.nef", "b.nef", "c.nef"])
        assert res.ids == {"a.nef": 101, "b.nef": 55, "c.nef": 103}
        assert set(res.duplicates) == {"b.nef"}
        assert fake.calls == [("import", {"paths": ["a.nef", "b.nef", "c.nef"], "mode": "add"})]

    def test_a_count_mismatch_refuses_to_guess(self):
        with pytest.raises(lm.LightCraftLibraryError, match=r"import returned 1 ids for 2 new files"):
            lm.import_files(FakeLib({"imported": [1], "duplicates": [], "failed": []}), ["a.nef", "b.nef"])

    def test_a_deleted_duplicate_is_refused(self):
        with pytest.raises(lm.LightCraftLibraryError, match=r"a.nef is in the library only as a deleted photo"):
            lm.import_files(FakeLib({"imported": [], "duplicates": [{"path": "a.nef", "existing": 9, "existingDeleted": True}], "failed": []}), ["a.nef"])

    def test_a_failed_file_is_refused(self):
        with pytest.raises(lm.LightCraftLibraryError, match=r"1 files failed to import"):
            lm.import_files(FakeLib({"imported": [7], "duplicates": [], "failed": [{"path": "b.nef", "reason": "unreadable"}]}), ["a.nef", "b.nef"])


class RenderLib:
    """A recording fake engine (the dependency of render_retouched). develop.auto and develop.reset change the history the way LightCraft
    does; 'export' writes the file after the call returns (on the first poll). gpu_states, when given, is replayed by cmd_app_gpu and makes
    this an engine started with gpu=True."""

    def __init__(self, history, out_bytes=b"jpegdata", iso=400, gpu_states=None, reset_raises=None, reset_leaves=None):
        self.history, self.out_bytes, self.calls, self.pending = list(history), out_bytes, [], None
        self.iso, self.reset_raises, self.reset_leaves = iso, reset_raises, reset_leaves
        self.gpu_states = list(gpu_states) if gpu_states is not None else None
        self.gpu = gpu_states is not None
        self.library = "lib-under-test"

    def call(self, tool, args):
        self.calls.append((tool, args))
        if tool == "cmd_photo_inspect":
            meta = {} if self.iso is None else {"iso": self.iso}
            return {"history": list(self.history), "meta": meta}
        if tool == "cmd_app_gpu":
            return self.gpu_states.pop(0)
        if tool == "command_run" and args["id"] == "develop.auto":
            self.history.append("Auto")
        if tool == "command_run" and args["id"] == "develop.reset":
            if self.reset_raises:
                raise self.reset_raises
            self.history.append(self.reset_leaves or "Reset")
        if tool == "export":
            self.pending = lm.Path(args["path"])
        return {}

    def sleep(self, _):
        if self.pending is not None and self.out_bytes:
            self.pending.write_bytes(self.out_bytes)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


@pytest.fixture(autouse=True)
def event_log(tmp_path, monkeypatch):
    """Every event this module records goes to a tmp file, never to the live settings.LIGHTCRAFT_EVENTS."""
    from bio_automation import settings

    path = tmp_path / "events" / "events.jsonl"
    monkeypatch.setattr(settings, "LIGHTCRAFT_EVENTS", path)
    monkeypatch.setattr(settings, "LIGHTCRAFT_LOG_DIR", tmp_path / "lclogs")
    return path


def events(path):
    from bio_automation.io import read_jsonl

    return [{k: v for k, v in e.items() if k != "time"} for e in read_jsonl(path)]


class TestRenderRetouched:
    def test_an_unedited_photo_gets_auto_then_its_own_edit_is_reset_and_verified(self, tmp_path, event_log):
        lib = RenderLib(history=[])
        out = tmp_path / "2025" / "2025-04-23" / "x.jpg"
        assert lm.render_retouched(lib, 7, out, sleep=lib.sleep, clock=Clock()) == {"owner_edited": False, "iso": 400, "warnings": []}
        assert [c[0] for c in lib.calls] == ["cmd_photo_inspect", "select_photos", "command_run", "export", "command_run", "cmd_photo_inspect"]
        assert lib.calls[2][1] == {"id": "develop.auto", "params": {}}
        assert lib.calls[3][1] == {"id": 7, "path": str(out), **lm.EXPORT_SETTINGS}
        assert lib.calls[4][1] == {"id": "develop.reset", "params": {"ids": [7]}}
        assert out.read_bytes() == b"jpegdata"
        assert not event_log.exists()

    def test_owner_edits_are_exported_as_they_are_and_never_reset(self, tmp_path):
        lib = RenderLib(history=["Exposure", "Crop"])
        assert lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())["owner_edited"] is True
        assert [c[0] for c in lib.calls] == ["cmd_photo_inspect", "export"]

    def test_a_history_ending_in_reset_counts_as_unedited(self, tmp_path):
        lib = RenderLib(history=["Exposure", "Reset"])
        assert lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())["owner_edited"] is False

    def test_no_file_raises_and_still_resets_its_own_edit(self, tmp_path):
        lib = RenderLib(history=[], out_bytes=b"")
        with pytest.raises(lm.LightCraftLibraryError, match=r"photo 7: no export at .*x\.jpg after 5 s"):
            lm.render_retouched(lib, 7, tmp_path / "x.jpg", wait_s=5, sleep=lib.sleep, clock=Clock())
        assert lib.calls[-2] == ("command_run", {"id": "develop.reset", "params": {"ids": [7]}})
        assert lib.history == ["Auto", "Reset"]

    def test_a_non_dict_inspect_is_refused_before_any_edit(self, tmp_path):
        class Odd(RenderLib):
            def call(self, tool, args):
                self.calls.append((tool, args))
                return "not a photo"

        lib = Odd(history=[])
        with pytest.raises(lm.LightCraftLibraryError, match=r"photo 7: photo\.inspect returned 'not a photo': cannot tell whether the owner edited it"):
            lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert [c[0] for c in lib.calls] == ["cmd_photo_inspect"]


class TestUnknownIso:
    def test_a_missing_iso_stays_unknown_and_is_said(self, tmp_path):
        lib = RenderLib(history=[], iso=None)
        res = lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert res == {"owner_edited": False, "iso": None,
                       "warnings": ["ISO unknown (photo.inspect meta.iso = None): high-ISO handling cannot be decided for this frame"]}

    def test_an_unreadable_iso_stays_unknown(self, tmp_path):
        lib = RenderLib(history=[], iso="n/a")
        res = lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert (res["iso"], res["warnings"]) == (None, ["ISO unknown (photo.inspect meta.iso = 'n/a'): high-ISO handling cannot be decided for this frame"])

    def test_a_known_iso_is_returned_as_an_int(self, tmp_path):
        lib = RenderLib(history=[], iso="3200")
        assert lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())["iso"] == 3200


class TestResetFailure:
    def test_a_reset_that_raises_is_recorded_and_raised(self, tmp_path, event_log):
        lib = RenderLib(history=[], reset_raises=RuntimeError("engine busy"))
        with pytest.raises(lm.LightCraftResetError, match=r"photo 7: develop\.reset failed \(RuntimeError: engine busy\); the library keeps"):
            lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert events(event_log) == [{"kind": "reset_failed", "photo_id": 7, "library": "lib-under-test", "error": "RuntimeError: engine busy"}]

    def test_a_reset_that_leaves_live_edits_is_recorded_and_raised(self, tmp_path, event_log):
        lib = RenderLib(history=[], reset_leaves="Exposure")         # the reset call returned, but the history does not end in Reset
        with pytest.raises(lm.LightCraftResetError, match=r"photo 7: still carries live edits after develop\.reset \(history \['Auto', 'Exposure'\]\)"):
            lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert events(event_log) == [{"kind": "reset_failed", "photo_id": 7, "library": "lib-under-test",
                                      "error": "history after reset: ['Auto', 'Exposure']"}]

    def test_a_reset_failure_after_a_failed_export_still_raises_the_reset_error_with_the_export_error_chained(self, tmp_path, event_log):
        lib = RenderLib(history=[], out_bytes=b"", reset_raises=RuntimeError("engine busy"))
        with pytest.raises(lm.LightCraftResetError, match=r"develop\.reset failed") as info:
            lm.render_retouched(lib, 7, tmp_path / "x.jpg", wait_s=2, sleep=lib.sleep, clock=Clock())
        assert isinstance(info.value.__context__.__context__, lm.LightCraftLibraryError)
        assert "no export at" in str(info.value.__context__.__context__)
        assert [e["kind"] for e in events(event_log)] == ["reset_failed"]


GPU_OK = {"enabled": True, "available": True, "adapter": "RTX 4070 Ti", "reason": None, "lastFallback": None}


class TestGpuFallback:
    def test_a_render_redone_on_the_cpu_is_deleted_recorded_and_raised(self, tmp_path, event_log):
        fell = {**GPU_OK, "lastFallback": {"photo": 7, "why": "out of device memory"}}
        lib = RenderLib(history=[], gpu_states=[GPU_OK, fell])
        out = tmp_path / "x.jpg"
        with pytest.raises(lm.LightCraftGpuError, match=r"photo 7: rendered on the CPU, not the GPU \(lastFallback \{'photo': 7, 'why': 'out of device memory'\}"):
            lm.render_retouched(lib, 7, out, sleep=lib.sleep, clock=Clock())
        assert not out.exists()                                  # a CPU render never stays where render_day would count it as done
        assert lib.history == ["Auto", "Reset"]                  # its own edit is still undone
        assert events(event_log) == [{"kind": "cpu_fallback", "photo_id": 7, "library": "lib-under-test", "out": str(out),
                                      "last_fallback": {"photo": 7, "why": "out of device memory"}, "enabled": True, "reason": None}]

    def test_a_fallback_from_an_earlier_render_is_not_blamed_on_this_one(self, tmp_path, event_log):
        old = {**GPU_OK, "lastFallback": {"photo": 3, "why": "out of device memory"}}
        lib = RenderLib(history=[], gpu_states=[old, old])
        res = lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert res["owner_edited"] is False and (tmp_path / "x.jpg").read_bytes() == b"jpegdata"
        assert not event_log.exists()

    def test_a_gpu_render_without_fallback_passes_and_checks_both_sides(self, tmp_path):
        lib = RenderLib(history=["Exposure"], gpu_states=[GPU_OK, GPU_OK])
        assert lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())["owner_edited"] is True
        assert [c[0] for c in lib.calls] == ["cmd_photo_inspect", "cmd_app_gpu", "export", "cmd_app_gpu"]

    def test_a_cpu_engine_never_asks_for_the_gpu_state(self, tmp_path):
        lib = RenderLib(history=["Exposure"])
        lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=lib.sleep, clock=Clock())
        assert [c[0] for c in lib.calls] == ["cmd_photo_inspect", "export"]


class TestCheckGpuState:
    def test_an_enabled_available_gpu_passes(self):
        assert lm.check_gpu_state(GPU_OK) is None

    def test_an_unavailable_gpu_is_refused_with_its_reason(self):
        with pytest.raises(lm.LightCraftGpuError, match=r"enabled=True, available=False, reason='no adapter'"):
            lm.check_gpu_state({**GPU_OK, "available": False, "reason": "no adapter"})


class ImmediateLib(RenderLib):
    """A headless export: the file is written before the call returns."""

    def call(self, tool, args):
        r = super().call(tool, args)
        if tool == "export":
            lm.Path(args["path"]).write_bytes(b"jpegdata")
        return r


def test_a_file_already_written_by_the_call_needs_no_waiting(tmp_path):
    lib = ImmediateLib(history=[])
    sleeps = []
    assert lm.render_retouched(lib, 7, tmp_path / "x.jpg", sleep=sleeps.append, clock=Clock())["owner_edited"] is False
    assert sleeps == []


FAKE_ENGINE = r'''
import json, os, sys
library = sys.argv[2]
sys.stderr.write("fake engine: warming up on " + library + "\n")
sys.stderr.flush()
gpu = json.loads(os.environ["FAKE_GPU_STATE"])
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "initialize":
        result = {}
    else:
        name = msg["params"]["name"]
        body = {"persistent": True, "path": library, "load": {"damaged": None}} if name == "cmd_library_info" else gpu
        result = {"content": [{"type": "text", "text": json.dumps(body)}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
'''


class TestEngineProcess:
    """A real child process speaking the engine's stdio protocol (python running a fake `mcp` script): stderr capture and the GPU start check."""

    @pytest.fixture()
    def engine(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.chdir(tmp_path)
        (tmp_path / "mcp").write_text(FAKE_ENGINE, encoding="utf-8")       # `<python> mcp --library X` runs this file
        return sys.executable

    def test_stderr_goes_to_a_log_under_the_cache_dir_and_a_gpu_engine_starts(self, tmp_path, monkeypatch, engine):
        import json

        monkeypatch.setenv("FAKE_GPU_STATE", json.dumps(GPU_OK))
        lib_dir = tmp_path / "scratch-1"
        with lm.LightCraftLibrary(lib_dir, engine, allow_gui=True, gpu=True) as lib:
            assert lib.gpu_at_start == GPU_OK
            assert lib.stderr_log == tmp_path / "lclogs" / "scratch-1.stderr.log"
        text = lib.stderr_log.read_text(encoding="utf-8")
        assert text.splitlines()[0].endswith(f"engine start: library={lib_dir} gpu=True")
        assert text.splitlines()[1] == f"fake engine: warming up on {lib_dir}"

    def test_a_gpu_engine_that_is_not_on_the_gpu_is_refused_at_start(self, tmp_path, monkeypatch, engine):
        import json

        monkeypatch.setenv("FAKE_GPU_STATE", json.dumps({**GPU_OK, "enabled": False, "reason": "LIGHTCRAFT_GPU=0"}))
        with pytest.raises(lm.LightCraftGpuError, match=r"not on the GPU: enabled=False, available=True, reason='LIGHTCRAFT_GPU=0'"):
            lm.LightCraftLibrary(tmp_path / "scratch-2", engine, allow_gui=True, gpu=True)

    def test_a_cpu_engine_is_not_asked_for_the_gpu(self, tmp_path, monkeypatch, engine):
        monkeypatch.setenv("FAKE_GPU_STATE", '"never read"')
        with lm.LightCraftLibrary(tmp_path / "scratch-3", engine, allow_gui=True, gpu=False) as lib:
            assert lib.gpu is False and not hasattr(lib, "gpu_at_start")
