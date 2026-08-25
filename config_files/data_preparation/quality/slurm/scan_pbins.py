#!/usr/bin/env python3
"""Finds packed files whose header or index is unusable.

Existence is not health: a .pbin left behind by an interrupted run can be megabytes on disk
and still report data_len=0, and --skip_existing then leaves it in place forever.
"""
import argparse
import sys
from pathlib import Path

from modalities.dataloader.create_packed_data import EmbeddedStreamData

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work_dir", type=Path, required=True, help="Blend working directory.")
parser.add_argument("--out", type=Path, default=None, help="Where to list the bad files.")
args = parser.parse_args()
W = args.work_dir / "packcfg"
bad = []
n = 0
for p in sorted(W.rglob("*.pbin")):
    n += 1
    try:
        s = EmbeddedStreamData(p, load_index=False)
        if s.data_len <= 0:
            bad.append((p, f"data_len={s.data_len}"))
            continue
    except Exception as e:
        bad.append((p, f"header: {type(e).__name__}"))
        continue
    if n % 5000 == 0:
        print(f"  scanned {n:,} ...", flush=True)
print(f"scanned {n:,} packed files, {len(bad)} unusable")
for path, why in bad:
    print(f"  BAD {path}  ({why})")
if args.out:
    args.out.write_text("\n".join(str(path) for path, _ in bad))
sys.exit(1 if bad else 0)
