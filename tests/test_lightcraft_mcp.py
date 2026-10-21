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
        assert lm.import_files(fake, ["a.nef", "b.nef", "c.nef"]) == {"a.nef": 101, "b.nef": 55, "c.nef": 103}
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
