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
    """Records every call; 'export' writes the file the way LightCraft does (after the call returns: on the first poll)."""

    def __init__(self, history, out_bytes=b"jpegdata"):
        self.history, self.out_bytes, self.calls, self.pending = history, out_bytes, [], None

    def call(self, tool, args):
        self.calls.append((tool, args))
        if tool == "cmd_photo_inspect":
            return {"history": self.history, "meta": {"iso": 400}}
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


class TestRenderRetouched:
    def test_an_unedited_photo_gets_auto_then_its_own_edit_is_reset(self, tmp_path):
        lib = RenderLib(history=[])
        out = tmp_path / "2025" / "2025-04-23" / "x.jpg"
        assert lm.render_retouched(lib, 7, out, sleep=lib.sleep, clock=Clock()) == {"owner_edited": False, "iso": 400}
        assert [c[0] for c in lib.calls] == ["cmd_photo_inspect", "select_photos", "command_run", "export", "command_run"]
        assert lib.calls[2][1] == {"id": "develop.auto", "params": {}}
        assert lib.calls[3][1] == {"id": 7, "path": str(out), **lm.EXPORT_SETTINGS}
        assert lib.calls[4][1] == {"id": "develop.reset", "params": {"ids": [7]}}
        assert out.read_bytes() == b"jpegdata"

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
        assert lib.calls[-1] == ("command_run", {"id": "develop.reset", "params": {"ids": [7]}})


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
