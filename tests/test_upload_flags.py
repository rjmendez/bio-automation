"""upload_flags: black-frame detection and manual notes appended to an upload set's observations.csv. Real JPEGs made with Pillow."""
from __future__ import annotations

import json

import pytest
from PIL import Image

from bio_automation import cli, upload_flags
from bio_automation.upload_flags import flag_observations_csv, jpeg_median_luma, load_manual_notes

HEADER = "group,files,status\r\n"


def make_set(tmp_path, groups):
    """groups: {group: {filename: grey level}} -> folder with the JPEGs and an observations.csv written CRLF like build_upload_set."""
    lines = [HEADER]
    for g, files in groups.items():
        for name, level in files.items():
            Image.new("RGB", (64, 48), (level, level, level)).save(tmp_path / name, "JPEG", quality=100)
        lines.append(f"{g},{';'.join(files)},pending\r\n")
    (tmp_path / "observations.csv").write_bytes("".join(lines).encode("utf-8"))
    return tmp_path


class TestLuma:
    def test_black_and_grey_jpegs(self, tmp_path):
        Image.new("RGB", (64, 48), (0, 0, 0)).save(tmp_path / "b.jpg", "JPEG", quality=100)
        Image.new("RGB", (64, 48), (100, 100, 100)).save(tmp_path / "g.jpg", "JPEG", quality=100)
        assert jpeg_median_luma(tmp_path / "b.jpg") == pytest.approx(0.0, abs=0.5)
        assert jpeg_median_luma(tmp_path / "g.jpg") == pytest.approx(100.0, abs=1.0)


class TestFlagCsv:
    def test_black_frame_is_flagged_with_its_name_and_other_groups_are_untouched(self, tmp_path):
        make_set(tmp_path, {1: {"g1_a.jpg": 0, "g1_b.jpg": 120, "g1_c.jpg": 0}, 2: {"g2_a.jpg": 90}})
        out = flag_observations_csv(tmp_path)
        assert out == {"groups": 2, "flagged_groups": 1, "black_frames": 2}
        assert (tmp_path / "observations.csv").read_bytes() == (
            "group,files,status,flags,flag_notes\r\n"
            '1,g1_a.jpg;g1_b.jpg;g1_c.jpg,pending,black_frame,"2 of 3 frames are black: g1_a.jpg, g1_c.jpg"\r\n'      # the comma makes the csv module quote it
            "2,g2_a.jpg,pending,,\r\n").encode("utf-8")

    def test_dim_but_not_black_frame_is_not_flagged(self, tmp_path):
        make_set(tmp_path, {1: {"a.jpg": 12}})
        assert flag_observations_csv(tmp_path)["black_frames"] == 0
        assert flag_observations_csv(tmp_path, black_luma=13.0)["black_frames"] == 1          # the threshold is the only difference

    def test_manual_flags_and_notes_combine_with_the_measured_one(self, tmp_path):
        make_set(tmp_path, {1: {"a.jpg": 0}, 2: {"b.jpg": 90}})
        flag_observations_csv(tmp_path, {1: {"flags": ["electronics"], "note": "circuit board"}, 2: {"flags": ["people_visible", "sign_or_text"], "note": ""}})
        lines = (tmp_path / "observations.csv").read_bytes().decode("utf-8").split("\r\n")
        assert lines[1] == "1,a.jpg,pending,black_frame;electronics,1 of 1 frames are black: a.jpg | circuit board"
        assert lines[2] == "2,b.jpg,pending,people_visible;sign_or_text,"

    def test_rerun_replaces_the_columns_instead_of_duplicating_them(self, tmp_path):
        make_set(tmp_path, {1: {"a.jpg": 0}})
        flag_observations_csv(tmp_path)
        first = (tmp_path / "observations.csv").read_bytes()
        flag_observations_csv(tmp_path)
        assert (tmp_path / "observations.csv").read_bytes() == first
        flag_observations_csv(tmp_path, {1: {"flags": ["pet"], "note": "dog"}})
        lines = (tmp_path / "observations.csv").read_bytes().decode("utf-8").split("\r\n")
        assert lines[0] == "group,files,status,flags,flag_notes"
        assert lines[1] == "1,a.jpg,pending,black_frame;pet,1 of 1 frames are black: a.jpg | dog"

    def test_notes_for_a_group_that_does_not_exist_are_refused_and_nothing_is_written(self, tmp_path):
        make_set(tmp_path, {1: {"a.jpg": 90}})
        before = (tmp_path / "observations.csv").read_bytes()
        with pytest.raises(ValueError, match=r"manual notes name groups that are not in observations.csv: \[7\]"):
            flag_observations_csv(tmp_path, {7: {"flags": ["x"], "note": ""}})
        assert (tmp_path / "observations.csv").read_bytes() == before

    def test_missing_csv(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"observations\.csv not found"):
            flag_observations_csv(tmp_path)


class TestManualNotes:
    def test_loads_and_keys_are_ints(self, tmp_path):
        p = tmp_path / "n.json"
        p.write_text(json.dumps({"92": {"flags": ["electronics"], "note": "board"}}), encoding="utf-8")
        assert load_manual_notes(p) == {92: {"flags": ["electronics"], "note": "board"}}
        assert load_manual_notes(None) == {}

    def test_bad_group_key(self, tmp_path):
        p = tmp_path / "n.json"
        p.write_text(json.dumps({"g92": {"flags": ["x"]}}), encoding="utf-8")
        with pytest.raises(ValueError, match="group key 'g92' is not a number"):
            load_manual_notes(p)

    def test_empty_flag_list(self, tmp_path):
        p = tmp_path / "n.json"
        p.write_text(json.dumps({"3": {"flags": []}}), encoding="utf-8")
        with pytest.raises(ValueError, match="group 3 needs a non-empty list of flag names"):
            load_manual_notes(p)


class TestCli:
    def test_command_prints_counts(self, tmp_path, capsys):
        make_set(tmp_path, {1: {"a.jpg": 0}, 2: {"b.jpg": 90}})
        args = cli.build_parser().parse_args(["flag-upload-set", str(tmp_path)])
        assert cli.cmd_flag_upload_set(args) == 0
        assert capsys.readouterr().out == '{"groups": 2, "flagged_groups": 1, "black_frames": 1}\n'

    def test_missing_csv_exits_with_the_message(self, tmp_path):
        args = cli.build_parser().parse_args(["flag-upload-set", str(tmp_path)])
        with pytest.raises(SystemExit, match=r"observations\.csv not found"):
            cli.cmd_flag_upload_set(args)
