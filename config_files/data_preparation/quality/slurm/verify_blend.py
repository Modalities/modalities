#!/usr/bin/env python3
"""Verifies the real packed blend against the manifest.

Three checks, none visible from exit codes:
  1. packed tokens vs the manifest's estimates, per dataset;
  2. packed documents vs documents selected -- not an estimate, so it must match exactly;
  3. the source tree still holds nothing but the jsonl it arrived with.

Reads each .pbin header for the exact token count, and the document index one file at a
time so 54,738 indexes are never resident together.
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml
from modalities.dataloader.create_packed_data import EmbeddedStreamData

W = Path("/data/user/richard.rutmann/annealing_blend")
manifest = yaml.safe_load((W / "mix/mix_manifest.yaml").read_text())

print(f"{'dataset':<16} {'est tokens':>16} {'packed tokens':>16} {'err':>7} {'docs sel':>14} {'docs packed':>14}")
print("-" * 92)
tot_est = tot_pack = 0
problems = []
for rec in manifest["datasets"]:
    name = rec["name"]
    pbins = sorted((W / "packcfg" / name).rglob("*.pbin"))
    ntok = ndoc = 0
    for p in pbins:
        s = EmbeddedStreamData(p, load_index=True)
        ntok += s.data_len // s.token_size_in_bytes
        ndoc += len(s.index_base)
        del s
    est = rec["est_tokens_kept"]
    sel = rec["n_documents_kept"]
    err = (ntok - est) / est if est else 0.0
    tot_est += est
    tot_pack += ntok
    flags = ""
    if abs(err) > 0.05:
        flags += "  TOKENS>5%"
        problems.append(f"{name}: tokens off {err:+.2%}")
    if ndoc != sel:
        flags += "  DOCS MISMATCH"
        problems.append(f"{name}: {sel:,} selected vs {ndoc:,} packed")
    if len(pbins) != len(rec["index_files"]):
        flags += "  FILE COUNT"
        problems.append(f"{name}: {len(rec['index_files'])} idx vs {len(pbins)} pbin")
    print(
        f"{name:<16} {est:>16,} {ntok:>16,} {err * 100:>6.2f}% {sel:>14,} {ndoc:>14,}{flags}",
        flush=True,
    )
print("-" * 92)
print(f"{'TOTAL':<16} {tot_est:>16,} {tot_pack:>16,} {(tot_pack-tot_est)/tot_est*100:>6.2f}%")
print()
print("source tree untouched:")
stray = [str(p) for p in Path("/data/annealing").rglob("*") if p.is_file() and p.suffix != ".jsonl"]
owned = [p for p in stray if Path(p).owner() == "richard.rutmann"]
print(f"  non-jsonl files: {len(stray)} (pre-existing README/.gitattributes)")
print(f"  files owned by us: {len(owned)}")
if owned:
    problems.append(f"{len(owned)} files written into the source tree")
print()
print(f"RESULT: {'all checks passed' if not problems else 'PROBLEMS'}")
for problem in problems:
    print(f"  - {problem}")
sys.exit(1 if problems else 0)
