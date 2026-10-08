"""Checks that a sidecar's byte offsets still describe the documents it claims.

The manifest in ``file_manifest`` catches a source tree that moved, by comparing paths
and sizes. That is cheap and covers what we have actually seen go wrong, but it is
circumstantial: a file can be rewritten at identical size. This module does the direct
test instead -- seek to a recorded offset, read the recorded length, and check that the
bytes there are the document the sidecar says they are.

Two details matter for the check to mean anything:

* Rows at ``byte_offset == 0`` are skipped. The first document of any JSONL file parses
  as JSON, so a row at offset 0 succeeds even when the file underneath is a completely
  different one. Testing only those rows is how a broken sidecar looks healthy.
* The comparison is against ``text_bytes``, not the join key. It works for every dataset
  including the two that have no identifier at all, and for the one whose key is a hash
  resolved out of a third corpus.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from modalities.dataloader.preprocessing.quality.file_manifest import FileManifest, ManifestError, load_manifest
from modalities.dataloader.preprocessing.quality.registry import DatasetEntry


@dataclass
class VerifyReport:
    """Outcome of verifying one dataset's sidecar.

    Attributes:
        dataset (str): Dataset name.
        n_parts (int): Sidecar parts present.
        n_sampled (int): Documents actually probed.
        n_readable (int): Probes whose byte range yielded parseable JSON.
        n_matching (int): Probes whose document had the recorded text length.
        manifest_problems (list[str]): Drift reported by the manifest, if there is one.
        has_manifest (bool): Whether a manifest was found.
        notes (list[str]): Anything else worth saying.
    """

    dataset: str
    n_parts: int = 0
    n_sampled: int = 0
    n_readable: int = 0
    n_matching: int = 0
    manifest_problems: list[str] = field(default_factory=list)
    has_manifest: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the sidecar can be trusted.

        Returns:
            bool: True if nothing drifted and every probe matched.
        """
        return not self.manifest_problems and self.n_sampled > 0 and self.n_matching == self.n_sampled

    @property
    def verdict(self) -> str:
        """A short label for the outcome.

        Returns:
            str: One of VALID, DRIFTED, BROKEN, PARTIAL, or EMPTY.
        """
        if self.n_sampled == 0:
            return "EMPTY"
        if self.manifest_problems:
            return "DRIFTED"
        if self.n_matching == self.n_sampled:
            return "VALID"
        if self.n_readable == 0:
            return "BROKEN"
        return "PARTIAL"


def verify_sidecar(
    sidecar_dir: Path,
    entry: DatasetEntry,
    n_parts: int = 8,
    n_rows_per_part: int = 4,
    seed: int = 0,
    check_manifest: bool = True,
) -> VerifyReport:
    """Probes a dataset's sidecar against its source files.

    Args:
        sidecar_dir (Path): The dataset's sidecar directory.
        entry (DatasetEntry): Registry entry supplying the source root.
        n_parts (int): How many sidecar parts to sample.
        n_rows_per_part (int): How many documents to probe per sampled part.
        seed (int): Sampling seed, so a verdict is reproducible.
        check_manifest (bool): Whether to compare the recorded file list against the
            source tree. Turn off only when deliberately verifying a sidecar whose
            manifest is expected to be stale, such as before adopting one.

    Returns:
        VerifyReport: What was probed and what matched.
    """
    sidecar_dir = Path(sidecar_dir)
    report = VerifyReport(dataset=entry.name)
    parts = sorted(sidecar_dir.glob("part-*.parquet"))
    report.n_parts = len(parts)
    if not parts:
        report.notes.append(f"no sidecar parts in {sidecar_dir}")
        return report

    manifest = load_manifest(sidecar_dir)
    report.has_manifest = manifest is not None
    if manifest is None:
        report.notes.append("no file manifest; file ids resolved by re-globbing, which is what this checks")
        if not entry.jsonl_root.exists():
            report.manifest_problems.append(f"source root {entry.jsonl_root} does not exist")
            return report
        source_files = entry.iter_files()
    else:
        if check_manifest:
            report.manifest_problems = manifest.drift(entry)
        source_files = manifest.resolve(entry)

    rnd = random.Random(seed)
    for part in rnd.sample(parts, min(n_parts, len(parts))):
        table = pq.read_table(part, columns=["file_id", "byte_offset", "byte_len", "text_bytes"])
        # Offset 0 parses in any JSONL file, so it is no evidence at all.
        candidates = [i for i in range(table.num_rows) if table.column("byte_offset")[i].as_py() > 0]
        if not candidates:
            continue
        for i in rnd.sample(candidates, min(n_rows_per_part, len(candidates))):
            report.n_sampled += 1
            file_id = table.column("file_id")[i].as_py()
            if file_id >= len(source_files):
                continue
            path = source_files[file_id]
            try:
                with open(path, "rb") as f:
                    f.seek(table.column("byte_offset")[i].as_py())
                    record = json.loads(f.read(table.column("byte_len")[i].as_py()))
            except (OSError, ValueError):
                continue
            report.n_readable += 1
            text = record.get(entry.text_field)
            if isinstance(text, str) and len(text.encode("utf-8")) == table.column("text_bytes")[i].as_py():
                report.n_matching += 1

    return report


def adopt_manifest(sidecar_dir: Path, entry: DatasetEntry, report: VerifyReport) -> Path:
    """Records a file manifest for a sidecar built before manifests existed.

    Only sensible once the sidecar has been verified, which is why the report is a
    required argument: stamping a manifest onto a stale sidecar would make a broken
    sidecar look pinned and trustworthy.

    Args:
        sidecar_dir (Path): The dataset's sidecar directory.
        entry (DatasetEntry): Registry entry supplying the current file list.
        report (VerifyReport): The verification that justifies adoption.

    Returns:
        Path: The written manifest path.

    Raises:
        ManifestError: If the report does not show the sidecar to be valid.
    """
    if not report.ok:
        raise ManifestError(
            f"refusing to write a manifest for {entry.name!r}: verification says {report.verdict} "
            f"({report.n_matching}/{report.n_sampled} probes matched). A manifest would make this "
            f"sidecar look pinned when its offsets do not describe the current files. Rebuild it."
        )
    return FileManifest.from_entry(entry).write(sidecar_dir)


def format_verify_report(reports: list[VerifyReport]) -> str:
    """Renders verification results as a table.

    Args:
        reports (list[VerifyReport]): One report per dataset.

    Returns:
        str: A printable report.
    """
    lines = [
        f"{'dataset':<18} {'parts':>7} {'probed':>7} {'read':>6} {'match':>6} {'manifest':>9}  verdict",
        "-" * 78,
    ]
    for r in reports:
        lines.append(
            f"{r.dataset:<18} {r.n_parts:>7} {r.n_sampled:>7} {r.n_readable:>6} {r.n_matching:>6} "
            f"{('yes' if r.has_manifest else 'no'):>9}  {r.verdict}"
        )

    broken = [r for r in reports if not r.ok]
    if broken:
        lines.append("")
        lines.append(f"{len(broken)} of {len(reports)} dataset(s) cannot be trusted:")
        for r in broken:
            lines.append(f"  {r.dataset} [{r.verdict}]")
            for problem in r.manifest_problems[:5]:
                lines.append(f"      {problem}")
            for note in r.notes:
                lines.append(f"      {note}")
            if r.n_sampled and r.n_readable == 0:
                lines.append("      every probed byte range was unreadable -- the source files changed")
        lines.append("")
        lines.append("Rebuild the sidecar, re-join and re-cube each of these before using them in a blend.")
    else:
        lines.append("")
        lines.append(f"all {len(reports)} dataset(s) verified: byte offsets describe the current source files")
    return "\n".join(lines)
