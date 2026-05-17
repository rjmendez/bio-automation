from __future__ import annotations

from pathlib import Path
import subprocess


def extract_exif(raw_path: Path) -> dict:
    result = subprocess.run(
        ["exiftool", "-j", str(raw_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    return json.loads(result.stdout)[0]

