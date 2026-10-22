"""A process that loops NasIoGate.acquire(stage) for a while against one shared gate file: the other half of tests/test_nas_io.py's
cross-process test. Not a test module. argv: db stage seconds hold_s out_json. Writes [{wait, enter, exit}, ...] (wall-clock seconds)."""
from __future__ import annotations

import json
import sys
import time

from bio_automation.nas_io import NasIoGate

db, stage, seconds, hold_s, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
gate = NasIoGate(db, 2, poll_s=0.01)
records = []
deadline = time.time() + seconds
while time.time() < deadline:
    with gate.acquire(stage) as waited:
        entered = time.time()
        time.sleep(hold_s)
        left = time.time()
    records.append({"wait": waited, "enter": entered, "exit": left})
with open(out, "w", encoding="utf-8") as f:
    json.dump(records, f)
