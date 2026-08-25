#!/usr/bin/env python3
"""Finds packed files whose header or index is unusable.

Existence is not health: a .pbin left behind by an interrupted run can be megabytes on disk
and still report data_len=0, and --skip_existing then leaves it in place forever.
"""
from pathlib import Path
import sys
from modalities.dataloader.create_packed_data import EmbeddedStreamData

W = Path("/data/user/richard.rutmann/annealing_blend/packcfg")
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
for p, why in bad:
    print(f"  BAD {p}  ({why})")
Path("/data/user/richard.rutmann/pack_probe/bad_pbins.txt").write_text(
    "\n".join(str(p) for p, _ in bad))
sys.exit(0)
