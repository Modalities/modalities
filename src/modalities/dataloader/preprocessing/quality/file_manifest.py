"""Pins the source file list a sidecar was built against.

A sidecar row locates its document by ``(file_id, byte_offset, byte_len)``, where
``file_id`` indexes the dataset's sorted file list. That list used to be re-derived from
the filesystem at every stage, which makes it a silent dependency on the source tree
never changing: rename a directory, re-shard a corpus, or let a transfer rewrite it, and
the same ``file_id`` names a different file. Every offset then points into the wrong
place, and nothing downstream can tell.

We learned this the expensive way. A transfer re-sharded four corpora after their
sidecars were built; ``Nemotron-CC`` went from 606 MB files to 137 MB files while keeping
almost the same file count, so the count check that existed passed and the byte offsets
pointed past the end of the file. Only a dataset whose file count dropped to zero failed
loudly.

So the file list is now recorded next to the sidecar and verified wherever it is used.
Drift becomes an error naming the file that moved, rather than a blend of garbage byte
ranges.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from modalities.dataloader.preprocessing.quality.registry import DatasetEntry

MANIFEST_NAME = "_files.json"


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, unreadable, or no longer matches the source."""


class SourceFile(BaseModel):
    """One source file, as it was when the sidecar was built.

    Attributes:
        file_id (int): Index of this file in the dataset's sorted file list. This is the
            value stored in the sidecar's ``file_id`` column.
        path (str): Path relative to the dataset's ``jsonl_root``. Relative so that a
            whole tree can be moved or a snapshot taken without invalidating the
            manifest, while the mapping from id to file stays exact.
        size (int): Size in bytes. Checked because it is what actually catches a
            re-sharded or partially transferred corpus, and unlike mtime it survives a
            copy that preserves content.
    """

    file_id: int
    path: str
    size: int


class FileManifest(BaseModel):
    """The source file list a sidecar was built against.

    Attributes:
        dataset (str): Dataset name, so a manifest cannot be read for the wrong dataset.
        jsonl_root (str): The root the paths are relative to, as recorded at build time.
            Informational: verification resolves against the *current* registry root, so
            moving the tree is fine and only the file set has to agree.
        glob (str): The pattern that produced the list, recorded for diagnosis.
        files (list[SourceFile]): One entry per file, ordered by ``file_id``.
    """

    dataset: str
    jsonl_root: str
    glob: str
    files: list[SourceFile] = Field(default_factory=list)

    @classmethod
    def from_entry(cls, entry: DatasetEntry) -> "FileManifest":
        """Records the dataset's current file list.

        Args:
            entry (DatasetEntry): The dataset to describe.

        Returns:
            FileManifest: A manifest of the files matching the entry right now.

        Raises:
            ManifestError: If the entry matches no files, which means a wrong root or
                glob rather than an empty corpus.
        """
        files = entry.iter_files()
        if not files:
            raise ManifestError(f"dataset {entry.name!r}: no files matched {entry.glob!r} under {entry.jsonl_root}")
        return cls(
            dataset=entry.name,
            jsonl_root=str(entry.jsonl_root),
            glob=entry.glob,
            files=[
                SourceFile(file_id=i, path=str(p.relative_to(entry.jsonl_root)), size=p.stat().st_size)
                for i, p in enumerate(files)
            ],
        )

    @staticmethod
    def path_for(sidecar_dir: Path) -> Path:
        """Names the manifest belonging to a sidecar directory.

        Args:
            sidecar_dir (Path): The dataset's sidecar directory.

        Returns:
            Path: Where the manifest lives.
        """
        return Path(sidecar_dir) / MANIFEST_NAME

    def write(self, sidecar_dir: Path) -> Path:
        """Writes the manifest beside the sidecar parts.

        Written atomically via a uniquely named temporary file, because the sidecar build
        is sharded across tasks that all describe the same file list and would otherwise
        race: a reader could observe a half-written manifest. Each task writes identical
        content, so last-writer-wins is correct.

        Args:
            sidecar_dir (Path): The dataset's sidecar directory. Created if absent.

        Returns:
            Path: The manifest path.
        """
        sidecar_dir = Path(sidecar_dir)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(sidecar_dir)
        tmp = target.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(self.model_dump_json(indent=2))
        os.replace(tmp, target)
        return target

    @classmethod
    def read(cls, sidecar_dir: Path) -> "FileManifest":
        """Loads the manifest belonging to a sidecar directory.

        Args:
            sidecar_dir (Path): The dataset's sidecar directory.

        Returns:
            FileManifest: The recorded file list.

        Raises:
            ManifestError: If the manifest is absent or unparseable.
        """
        path = cls.path_for(sidecar_dir)
        if not path.exists():
            raise ManifestError(
                f"no source file manifest at {path}. This sidecar was built before file lists were "
                f"recorded, so its byte offsets cannot be checked against the source tree. Run "
                f"'modalities quality verify-sidecar --adopt' to verify it and stamp a manifest, or "
                f"rebuild the sidecar."
            )
        try:
            return cls.model_validate(json.loads(path.read_text()))
        except Exception as e:
            raise ManifestError(f"cannot read source file manifest {path}: {e}") from e

    def resolve(self, entry: DatasetEntry) -> list[Path]:
        """Maps recorded file ids to paths under the entry's current root.

        Resolution is by recorded *path*, not by re-globbing, so a file added to or
        removed from the source tree cannot shift the mapping.

        Args:
            entry (DatasetEntry): The dataset, supplying the current root.

        Returns:
            list[Path]: Absolute paths indexed by file id.
        """
        root = Path(entry.jsonl_root)
        return [root / f.path for f in sorted(self.files, key=lambda f: f.file_id)]

    def drift(self, entry: DatasetEntry, check_sizes: bool = True) -> list[str]:
        """Describes how the source tree differs from what was recorded.

        Args:
            entry (DatasetEntry): The dataset to check, supplying the current root.
            check_sizes (bool): Whether to compare file sizes. Sizes are what catch a
                re-sharded or half-transferred corpus, so this is on by default.

        Returns:
            list[str]: One human-readable line per problem, empty if the tree agrees.
                Truncated to the first 20 problems plus a count, since a re-shard makes
                every file differ and a wall of text helps nobody.
        """
        problems: list[str] = []
        root = Path(entry.jsonl_root)
        if not root.exists():
            return [f"source root {root} does not exist"]

        for f in sorted(self.files, key=lambda f: f.file_id):
            path = root / f.path
            if not path.exists():
                problems.append(f"file_id {f.file_id}: {f.path} is gone")
                continue
            if check_sizes:
                size = path.stat().st_size
                if size != f.size:
                    problems.append(
                        f"file_id {f.file_id}: {f.path} is {size:,} bytes, was {f.size:,} when the sidecar was built"
                    )

        recorded = {f.path for f in self.files}
        current = {str(p.relative_to(root)) for p in entry.iter_files()}
        added = sorted(current - recorded)
        if added:
            problems.append(
                f"{len(added)} file(s) added to the source tree since the sidecar was built, e.g. "
                f"{added[:3]} -- they hold no sidecar rows and will not take part in the blend"
            )

        if len(problems) > 20:
            return problems[:20] + [f"... and {len(problems) - 20} further problems"]
        return problems

    def require_current(self, entry: DatasetEntry, check_sizes: bool = True) -> list[Path]:
        """Resolves file ids to paths, refusing if the source tree has drifted.

        Args:
            entry (DatasetEntry): The dataset to check and resolve against.
            check_sizes (bool): Whether to compare file sizes.

        Returns:
            list[Path]: Absolute paths indexed by file id.

        Raises:
            ManifestError: If the source tree no longer matches the manifest. Byte
                offsets recorded against the old tree do not describe the new one, so
                continuing would produce a blend of wrong documents.
        """
        problems = self.drift(entry, check_sizes=check_sizes)
        if problems:
            listed = "\n".join(f"  {p}" for p in problems)
            raise ManifestError(
                f"dataset {entry.name!r}: the source tree changed since the sidecar was built, so "
                f"its byte offsets no longer describe these files:\n{listed}\n"
                f"Rebuild this dataset's sidecar (and re-join and re-cube it) before using it."
            )
        return self.resolve(entry)


def load_manifest(sidecar_dir: Path) -> Optional[FileManifest]:
    """Reads a manifest if one is present.

    Args:
        sidecar_dir (Path): The dataset's sidecar directory.

    Returns:
        Optional[FileManifest]: The manifest, or None if there is none.
    """
    if not FileManifest.path_for(sidecar_dir).exists():
        return None
    return FileManifest.read(sidecar_dir)
