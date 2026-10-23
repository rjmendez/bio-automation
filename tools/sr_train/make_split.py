"""Reproduce aws_train.py's exact deterministic split (sorted, shuffle(seed=1234),
first N held out) over the uploaded 1500-file sample, so the patches precomputed
locally use the identical train/held-out boundary the AWS run will use."""
import random
import sys
from pathlib import Path

names_file = Path(sys.argv[1])   # the flat filename list already used for the S3 upload
source_dir = Path(sys.argv[2])   # local F: path these filenames live under
held_out_n = int(sys.argv[3])
out_train = Path(sys.argv[4])
out_held_out = Path(sys.argv[5])

names = sorted(n.strip() for n in names_file.read_text().splitlines() if n.strip())
rng = random.Random(1234)
rng.shuffle(names)
held_out, train = names[:held_out_n], names[held_out_n:]

out_train.write_text("\n".join(str(source_dir / n) for n in train))
out_held_out.write_text("\n".join(str(source_dir / n) for n in held_out))
print(f"train: {len(train)}  held_out: {len(held_out)}")
