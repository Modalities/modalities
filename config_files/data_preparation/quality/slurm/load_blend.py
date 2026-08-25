#!/usr/bin/env python3
"""Final check: the packed blend actually loads and serves samples.

Everything else verifies files on disk. This is the only check that exercises the path
training will take -- WeightedCombinedDataset over the packed files, with the manifest's
repeat factors, pulling samples at the boundaries and the middle where an off-by-one in the
affine permutation would show.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import yaml

from modalities.dataloader.dataset import PackedMemMapDatasetContinuous, WeightedCombinedDataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work_dir", type=Path, required=True, help="Blend working directory.")
parser.add_argument("--sequence_length", type=int, default=2048, help="Block size.")
args = parser.parse_args()
W = args.work_dir
SEQ = args.sequence_length
manifest = yaml.safe_load((W / "mix/mix_manifest.yaml").read_text())

t0 = time.time()
datasets, factors = [], []
for rec in manifest["datasets"]:
    for pbin in sorted((W / "packcfg" / rec["name"]).rglob("*.pbin")):
        datasets.append(PackedMemMapDatasetContinuous(
            raw_data_path=pbin, sample_key="input_ids", block_size=SEQ, reuse_last_target=True))
        factors.append(float(rec["ratio"]))
print(f"  opened {len(datasets):,} packed files in {time.time()-t0:.0f}s")
print(f"  distinct repeat factors: {sorted(set(factors))}")

blend = WeightedCombinedDataset(datasets=datasets, repeat_factors=factors, seed=42)
expected = sum(int(len(d) * f) for d, f in zip(datasets, factors))
print(f"  blend length {len(blend):,} samples of {SEQ} tokens  (expected ~{expected:,})")
print(f"  = {len(blend)*SEQ/1e12:.3f} T tokens per epoch over the blend")

bad = []
for i in [0, 1, len(blend)//4, len(blend)//2, 3*len(blend)//4, len(blend)-2, len(blend)-1]:
    s = blend[i]["input_ids"]
    if len(s) != SEQ:
        bad.append(f"sample {i}: {len(s)} tokens")
print(f"  pulled 7 samples across the range, all {SEQ} tokens" if not bad else f"  BAD: {bad}")

frac = [f for f in factors if f != int(f)]
print(f"  fractional factors exercised: {sorted(set(frac))}" if frac else "  no fractional factors")
sys.exit(1 if bad else 0)
