#!/usr/bin/env python3
"""Packs many configs in one process, loading the tokenizer once.

`modalities data pack_encoded_data` builds its components -- tokenizer included -- from the
config on every call, so driving it once per source file pays the tokenizer load every time.
Measured on a compute node that load is 24.7 s, against ~3 s of actual work for a Dolmino
file. With 54,738 configs, of which 40,003 are Dolmino, that is roughly 375 core-hours of
startup for about 48 core-hours of tokenising: the overhead is eight times the work.

This driver loads the tokenizer once and constructs a `PackedDataGenerator` per config,
which is what `pack_encoded_data` does internally anyway. Everything else -- the index, the
jq pattern, the worker count -- still comes from the rendered config, so the output is
identical to running the CLI per file.

Takes a slice of the config list so it can run as a SLURM array.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from modalities.config.config import load_app_config_dict
from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer


def _marker_for(destination: Path) -> Path:
    """The file recording which fingerprint an output was produced from.

    Returns:
        Path: ``<destination>.fingerprint``.
    """
    return destination.with_name(destination.name + ".fingerprint")


def _fingerprint_of(config_path: Path) -> str | None:
    """The fingerprint `write-packing-configs` recorded for this job.

    Returns:
        str | None: The digest, or None when the config predates fingerprinting.
    """
    try:
        return config_path.with_suffix(".fingerprint").read_text().strip()
    except OSError:
        return None


def _is_usable(destination: Path, fingerprint: str | None) -> bool:
    """Whether an existing packed file can be left alone.

    Existence is not health, and health is not currency. A .pbin from an interrupted run
    can be megabytes on disk and still report ``data_len=0``; skipping on existence alone
    left exactly one such file in the blend, and it only surfaced during final
    verification. Separately, a changed selection rewrites an index under the same name,
    so a structurally fine output can hold the documents the *previous* selection chose --
    hence the fingerprint check as well.

    Args:
        destination (Path): The packed file to check.
        fingerprint (str | None): The fingerprint this job should have been packed from.

    Returns:
        bool: True if the header reads, declares a non-empty data section, and the
            recorded fingerprint matches.
    """
    if not destination.exists():
        return False
    if fingerprint is not None:
        marker = _marker_for(destination)
        try:
            if marker.read_text().strip() != fingerprint:
                return False
        except OSError:
            return False
    try:
        return EmbeddedStreamData(destination, load_index=False).data_len > 0
    except Exception:
        return False


def _pack_to(destination: Path, fingerprint: str | None, run) -> None:
    """Packs into a temporary file and publishes it, so a failure leaves nothing usable.

    Two problems this avoids. `PackedDataGenerator.run` refuses a destination that already
    exists, so a damaged .pbin could never be replaced -- every retry raised "file already
    exists" instead of repacking it. And an interrupted rebuild used to leave the previous
    run's still-matching fingerprint beside a half-written file, which the next run would
    then accept: the header reads, the data length is non-zero, and the record agrees.

    So the record is torn up before the attempt and rewritten only after the output is
    fully in place.

    Args:
        destination (Path): Where the finished .pbin belongs.
        fingerprint (str | None): The fingerprint to record on success.
        run (Callable[[Path], None]): Packs into the path it is given.
    """
    marker = _marker_for(destination)
    marker.unlink(missing_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        run(partial)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if fingerprint is not None:
        marker.write_text(fingerprint)


def main() -> int:
    """Packs this task's slice of the config list.

    Returns:
        int: Process exit status; non-zero if any config failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_list", type=Path, required=True, help="File of packing config paths.")
    parser.add_argument("--shard_id", type=int, required=True, help="This task's index.")
    parser.add_argument("--num_shards", type=int, required=True, help="Total tasks sharing the list.")
    parser.add_argument("--tokenizer_config", type=Path, required=True, help="Config holding the tokenizer section.")
    parser.add_argument("--skip_existing", action="store_true", help="Leave already-packed outputs alone.")
    args = parser.parse_args()

    paths = [Path(line) for line in args.config_list.read_text().split() if line]
    # Strided rather than contiguous: the list is grouped by dataset, so contiguous slices
    # would give one task all of Dolmino's small files and another all of HPLT's large ones.
    mine = paths[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)} of {len(paths)} configs", flush=True)

    tokenizer_section = load_app_config_dict(args.tokenizer_config)["tokenizer"]["config"]
    start = time.time()
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=tokenizer_section["pretrained_model_name_or_path"],
        padding=tokenizer_section.get("padding", False),
        truncation=tokenizer_section.get("truncation", False),
    )
    print(f"tokenizer loaded once in {time.time() - start:.1f}s", flush=True)

    packed = skipped = failed = 0
    t0 = time.time()
    for i, config_path in enumerate(mine):
        settings = load_app_config_dict(config_path)["settings"]
        destination = Path(settings["dst_path"])
        fingerprint = _fingerprint_of(config_path)
        if args.skip_existing and _is_usable(destination, fingerprint):
            skipped += 1
            continue
        try:
            # load_app_config_dict returns plain strings; the component factory normally
            # coerces these to Path via pydantic, and PackedDataGenerator calls .is_file().
            _pack_to(
                destination,
                fingerprint,
                lambda target: PackedDataGenerator(
                    Path(settings["src_path"]),
                    tokenizer=tokenizer,
                    eod_token=settings["eod_token"],
                    number_of_processes=settings["num_cpus"],
                    jq_pattern=settings["jq_pattern"],
                    processing_batch_size=settings["processing_batch_size"],
                    raw_samples_queue_size=settings["raw_samples_queue_size"],
                    processed_samples_queue_size=settings["processed_samples_queue_size"],
                    index_path=Path(settings["index_path"]) if settings.get("index_path") else None,
                ).run(target),
            )
            packed += 1
        except Exception as e:  # keep going; one bad file must not lose the whole slice
            failed += 1
            print(f"FAILED {config_path}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(
                f"  {i + 1}/{len(mine)} at {rate:.2f} configs/s, "
                f"eta {(len(mine) - i - 1) / rate / 60:.0f} min",
                flush=True,
            )

    print(f"shard {args.shard_id}: packed {packed}, skipped {skipped}, failed {failed}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
