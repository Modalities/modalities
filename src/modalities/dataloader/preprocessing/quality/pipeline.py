"""Whole-blend orchestration of the quality selection stages.

Each function here drives one stage across every dataset of a registry and leaves its
output in a fixed place under a working directory, so the stages can be run
independently, re-run for a single dataset, and resumed after a failure:

``<work_dir>/calibration.yaml``   token estimator constants, one entry per dataset
``<work_dir>/sidecar/<dataset>/`` per-document parquet parts
``<work_dir>/buckets/<split>/``   annotation shards partitioned for joining
``<work_dir>/cube/<dataset>.parquet``  aggregated counts, read by the preview
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from modalities.dataloader.preprocessing.quality.annotation_join import (
    JoinReport,
    bucket_annotations,
    join_annotations,
    read_bucket_metadata,
)
from modalities.dataloader.preprocessing.quality.cube import Cube, build_cube
from modalities.dataloader.preprocessing.quality.materialize import materialize_blend
from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry, KeyKind
from modalities.dataloader.preprocessing.quality.selection import (
    BlendResult,
    SelectionConfig,
    evaluate_blend,
    SelectionError,
    format_blend_report,
    predicate_breakdown,
)
from modalities.dataloader.preprocessing.quality.sidecar import SidecarBuilder, resolve_source_pointers
from modalities.dataloader.preprocessing.quality.tokens import CalibrationSet, calibrate_dataset
from modalities.dataloader.preprocessing.quality.verify import VerifyReport, adopt_manifest, verify_sidecar
from modalities.utils.logger_utils import get_logger


def calibration_path(work_dir: Path) -> Path:
    """Location of the calibration file within a working directory.

    Args:
        work_dir (Path): The working directory.

    Returns:
        Path: Path to ``calibration.yaml``.
    """
    return Path(work_dir) / "calibration.yaml"


def sidecar_dir(work_dir: Path, dataset_name: str) -> Path:
    """Location of one dataset's sidecar parts.

    Args:
        work_dir (Path): The working directory.
        dataset_name (str): The dataset name.

    Returns:
        Path: Directory holding that dataset's parquet parts.
    """
    return Path(work_dir) / "sidecar" / dataset_name


def cube_path(work_dir: Path, dataset_name: str) -> Path:
    """Location of one dataset's cube.

    Args:
        work_dir (Path): The working directory.
        dataset_name (str): The dataset name.

    Returns:
        Path: Path to that dataset's cube parquet.
    """
    return Path(work_dir) / "cube" / f"{dataset_name}.parquet"


def bucket_dir(work_dir: Path, split: str) -> Path:
    """Location of one annotation split's buckets.

    Args:
        work_dir (Path): The working directory.
        split (str): The annotation split path.

    Returns:
        Path: Directory holding that split's bucket files.
    """
    return Path(work_dir) / "buckets" / split.replace("/", "__")


def calibrate_blend(
    registry: CorpusRegistry,
    work_dir: Path,
    tokenizer,
    tokenizer_name: str,
    sample_size: int = 2000,
    only: Optional[list[str]] = None,
) -> CalibrationSet:
    """Measures token-estimation constants for every dataset.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        work_dir (Path): Working directory receiving ``calibration.yaml``.
        tokenizer: The tokenizer training will use.
        tokenizer_name (str): Identifier recorded with each measurement.
        sample_size (int): Documents to tokenize per dataset.
        only (Optional[list[str]]): Restrict to these dataset names, merging the result
            into any existing calibration file.

    Returns:
        CalibrationSet: The calibrations, also written to ``calibration.yaml``.
    """
    path = calibration_path(work_dir)
    existing = CalibrationSet.from_yaml(path) if path.is_file() else CalibrationSet(tokenizer=tokenizer_name)
    if existing.tokenizer != tokenizer_name:
        get_logger(name="main").warning(
            f"existing calibration was measured with {existing.tokenizer!r}, now measuring with "
            f"{tokenizer_name!r}; entries for other datasets are stale and should be re-measured"
        )
        existing.tokenizer = tokenizer_name

    for dataset in registry.enabled_datasets():
        if only and dataset.name not in only:
            continue
        calibration = calibrate_dataset(
            dataset_name=dataset.name,
            file_paths=dataset.iter_files(),
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            text_field=dataset.text_field,
            sample_size=sample_size,
        )
        existing.calibrations[dataset.name] = calibration
        # Written after every dataset, not once at the end, so interrupting the stage
        # keeps what it already measured. Re-running with `--only` then fills the rest.
        existing.to_yaml(path)
        get_logger(name="main").info(
            f"{dataset.name}: {calibration.bytes_per_token:.3f} bytes/token"
            + (
                f", using native field {calibration.native_field!r} scaled by {calibration.native_scale:.4f}"
                if calibration.uses_native_field()
                else ""
            )
        )
    return existing


def plan_sidecar_work(
    registry: CorpusRegistry,
    only: Optional[list[str]] = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, list[int]]:
    """Assigns each task its share of the per-file sidecar work.

    The unit of work is one JSONL file, and the work list is flattened across every
    selected dataset before being divided, so one array covers the whole blend however
    unevenly the file counts fall -- and they fall very unevenly, from four files to
    forty thousand.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        only (Optional[list[str]]): Restrict to these dataset names.
        shard_id (int): This task's index in ``[0, num_shards)``.
        num_shards (int): Total number of tasks.

    Returns:
        dict[str, list[int]]: File ids this task should build, per dataset. Datasets with
            nothing for this task are absent.

    Raises:
        ValueError: If the shard selection is out of range.
    """
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id {shard_id} is not in [0, {num_shards})")

    work: list[tuple[str, int]] = []
    for dataset in registry.enabled_datasets():
        if only and dataset.name not in only:
            continue
        work.extend((dataset.name, file_id) for file_id in range(len(dataset.iter_files())))

    assigned: dict[str, list[int]] = {}
    # Strided, so no task ends up holding only the largest dataset's files.
    for position, (name, file_id) in enumerate(work):
        if position % num_shards == shard_id:
            assigned.setdefault(name, []).append(file_id)
    return assigned


def build_sidecars(
    registry: CorpusRegistry,
    work_dir: Path,
    only: Optional[list[str]] = None,
    index_root: Optional[Path] = None,
    file_ids: Optional[list[int]] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    show_progress: bool = True,
) -> dict[str, int]:
    """Builds the per-document table for every dataset.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        work_dir (Path): Working directory receiving ``sidecar/<dataset>/``.
        only (Optional[list[str]]): Restrict to these dataset names.
        index_root (Optional[Path]): Where JSONL index files live or should be created.
            Defaults to ``work_dir/idx``, never the source tree. Source corpora are
            typically shared and read-only, and an index written beside a JSONL file is a
            modification of somebody else's data; pass an explicit path to override.
        file_ids (Optional[list[int]]): Restrict to these file ids explicitly. Applies to
            every selected dataset and cannot be combined with sharding.
        shard_id (int): This task's index in ``[0, num_shards)``.
        num_shards (int): Total number of tasks sharing the work.
        show_progress (bool): Whether to show progress bars.

    Returns:
        dict[str, int]: Documents written by this task, per dataset.

    Raises:
        ValueError: If explicit file ids are combined with a shard selection, since the
            two express the same thing and the outcome would depend on which won.
    """
    if file_ids is not None and num_shards != 1:
        raise ValueError("pass either explicit file_ids or a shard selection, not both")

    # Never default to writing indexes beside the source JSONL, which is what
    # SidecarBuilder does when given no index root.
    index_root = Path(index_root) if index_root is not None else Path(work_dir) / "idx"

    calibrations = CalibrationSet.from_yaml(calibration_path(work_dir))
    selected = [d for d in registry.enabled_datasets() if not only or d.name in only]
    if file_ids is not None:
        assignment: dict[str, Optional[list[int]]] = {d.name: file_ids for d in selected}
    elif num_shards == 1:
        assignment = {d.name: None for d in selected}
    else:
        assignment = plan_sidecar_work(registry, only=only, shard_id=shard_id, num_shards=num_shards)
        get_logger(name="main").info(
            f"shard {shard_id}/{num_shards} builds "
            + (", ".join(f"{name}:{len(ids)} file(s)" for name, ids in sorted(assignment.items())) or "nothing")
        )

    written: dict[str, int] = {}
    for dataset in selected:
        if dataset.name not in assignment:
            continue
        builder = SidecarBuilder(
            dataset=dataset,
            calibration=calibrations.get(dataset.name),
            index_root=index_root / dataset.name,
        )
        parts = builder.build(
            sidecar_dir(work_dir, dataset.name),
            file_ids=assignment[dataset.name],
            show_progress=show_progress,
        )
        written[dataset.name] = sum(parts.values())

        # Safe to run per task: each task owns the parts it just wrote.
        if dataset.key is not None and dataset.key.kind == KeyKind.SOURCE_POINTER:
            n_resolved = resolve_source_pointers(
                sidecar_dir(work_dir, dataset.name), dataset, only_parts=assignment[dataset.name]
            )
            get_logger(name="main").info(
                f"{dataset.name}: resolved {n_resolved:,} of {written[dataset.name]:,} pointers into source-corpus keys"
            )
    return written


def bucket_blend_annotations(
    registry: CorpusRegistry,
    work_dir: Path,
    only: Optional[list[str]] = None,
    n_buckets: int = 1024,
    shard_id: int = 0,
    num_shards: int = 1,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, int]:
    """Partitions every annotation split the blend needs, ready for joining.

    The expensive stage of the join and the one worth parallelising. Splits shared by
    several datasets are bucketed once.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        work_dir (Path): Working directory receiving ``buckets/<split>/``.
        only (Optional[list[str]]): Restrict to the splits these datasets need.
        n_buckets (int): Partitions per split.
        shard_id (int): This task's index in ``[0, num_shards)``.
        num_shards (int): How many tasks bucket each split.
        force (bool): Re-bucket a split whose output is already complete.
        show_progress (bool): Whether to show progress bars.

    Returns:
        dict[str, int]: Rows written by this task, per split.
    """
    splits: dict[str, Optional[str]] = {}
    for dataset in registry.enabled_datasets():
        if only and dataset.name not in only:
            continue
        if dataset.annotation_split and dataset.annotation_split not in splits:
            # Whether keys need normalising is a property of the split's key space, so
            # the first dataset naming a split settles it for every other user of it.
            splits[dataset.annotation_split] = "urn_uuid" if dataset.key.kind == KeyKind.URN_UUID_FIELD else None

    written: dict[str, int] = {}
    for split, normalize in splits.items():
        shards = registry.annotation_shards(split)
        if not shards:
            get_logger(name="main").warning(f"split {split!r}: no shards on disk, nothing to bucket")
            continue
        out_dir = bucket_dir(work_dir, split)
        if not force:
            try:
                meta = read_bucket_metadata(out_dir)
            except Exception:
                pass
            else:
                get_logger(name="main").info(
                    f"split {split}: already bucketed ({meta['n_rows']:,} rows, {meta['n_buckets']} buckets), skipping"
                )
                continue
        n_rows, columns = bucket_annotations(
            shard_paths=shards,
            out_dir=out_dir,
            n_buckets=n_buckets,
            normalize_key=normalize,
            shard_id=shard_id,
            num_shards=num_shards,
            show_progress=show_progress,
        )
        written[split] = n_rows
        get_logger(name="main").info(
            f"split {split}: shard {shard_id}/{num_shards} wrote {n_rows:,} rows "
            f"from {len(shards)} input shard(s), columns {columns}"
        )
    return written


def join_blend_annotations(
    registry: CorpusRegistry,
    work_dir: Path,
    only: Optional[list[str]] = None,
    resume: bool = False,
    show_progress: bool = True,
) -> list[JoinReport]:
    """Attaches the bucketed annotations to every annotated dataset's sidecar.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        work_dir (Path): Working directory holding the sidecars and the buckets.
        only (Optional[list[str]]): Restrict to these dataset names.
        resume (bool): Skip sidecar parts that already carry labels, to continue an
            interrupted run.
        show_progress (bool): Whether to show progress bars.

    Returns:
        list[JoinReport]: One report per joined dataset.
    """
    reports: list[JoinReport] = []
    for dataset in registry.enabled_datasets():
        if only and dataset.name not in only:
            continue
        if not dataset.annotation_split:
            continue

        buckets = bucket_dir(work_dir, dataset.annotation_split)
        if not buckets.is_dir():
            get_logger(name="main").warning(
                f"{dataset.name}: split {dataset.annotation_split!r} has not been bucketed; its documents stay "
                "unannotated and any predicate on them falls back to the missing-annotation policy"
            )
            continue

        reports.append(
            join_annotations(
                sidecar_dir=sidecar_dir(work_dir, dataset.name),
                annotation_bucket_dir=buckets,
                dataset_name=dataset.name,
                split_name=dataset.annotation_split,
                resume=resume,
                show_progress=show_progress,
            )
        )

    # One file per dataset. A single shared join_report.json meant 16 parallel `--only`
    # tasks overwrote each other and only the last one's coverage survived.
    report_dir = Path(work_dir) / "join_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    for r in reports:
        (report_dir / f"{r.dataset}.json").write_text(json.dumps(r.to_dict(), indent=1))
    # Merged view, rebuilt from whatever per-dataset files exist so far.
    merged = [json.loads(p.read_text()) for p in sorted(report_dir.glob("*.json"))]
    (Path(work_dir) / "join_report.json").write_text(json.dumps(merged, indent=1))
    return reports


def verify_sidecars(
    registry: CorpusRegistry,
    work_dir: Path,
    only: Optional[list[str]] = None,
    n_parts: int = 8,
    n_rows_per_part: int = 4,
    adopt: bool = False,
) -> list[VerifyReport]:
    """Checks every dataset's sidecar against its source files.

    Worth running before ``apply`` on any blend whose source tree might have been touched
    since the sidecars were built, and after any data transfer. A sidecar whose corpus was
    re-sharded underneath it produces a blend of wrong byte ranges, and this is the only
    stage that looks at the source bytes to find out.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        work_dir (Path): Working directory holding the sidecars.
        only (Optional[list[str]]): Restrict to these dataset names.
        n_parts (int): Sidecar parts to sample per dataset.
        n_rows_per_part (int): Documents to probe per sampled part.
        adopt (bool): Write a file manifest for verified sidecars that lack one, so later
            stages can check for drift cheaply. Only verified sidecars are stamped.

    Returns:
        list[VerifyReport]: One report per dataset, in registry order.
    """
    datasets = [d for d in registry.enabled_datasets() if only is None or d.name in set(only)]
    reports: list[VerifyReport] = []
    for dataset in datasets:
        directory = sidecar_dir(work_dir, dataset.name)
        report = verify_sidecar(directory, dataset, n_parts=n_parts, n_rows_per_part=n_rows_per_part)
        if adopt and report.ok and not report.has_manifest:
            adopt_manifest(directory, dataset, report)
            report.has_manifest = True
            report.notes.append("manifest adopted after verification")
        reports.append(report)
    return reports


def build_cubes(
    registry: CorpusRegistry,
    work_dir: Path,
    only: Optional[list[str]] = None,
    n_score_bins: int = 10,
    label_dimensions: Optional[list[str]] = None,
) -> dict[str, Cube]:
    """Aggregates every dataset's sidecar into a cube.

    Args:
        registry (CorpusRegistry): The blend's datasets.
        work_dir (Path): Working directory holding sidecars and receiving cubes.
        only (Optional[list[str]]): Restrict to these dataset names.
        n_score_bins (int): Quantile bins per native metric.
        label_dimensions (Optional[list[str]]): Annotation columns to group on. Defaults to
            :data:`~...cube.DEFAULT_LABEL_DIMENSIONS`, which is a subset of the columns the
            join attaches -- grouping on all twelve would multiply the cell count by
            thousands. Name a field here when a selection needs to threshold on it, or the
            preview will have to scan the sidecar instead.

    Returns:
        dict[str, Cube]: The cubes, also written under ``cube/``.
    """
    cubes: dict[str, Cube] = {}
    failures: list[tuple[str, Exception]] = []
    for dataset in registry.enabled_datasets():
        if only and dataset.name not in only:
            continue
        directory = sidecar_dir(work_dir, dataset.name)
        if not directory.is_dir():
            get_logger(name="main").warning(f"{dataset.name}: no sidecar at {directory}, skipping cube")
            continue
        # One unbuildable dataset must not cost the others their cubes. A single failure
        # used to abort the stage: nine cubes were written and six perfectly healthy
        # datasets were never attempted.
        try:
            cube = build_cube(
                directory,
                dataset.name,
                n_score_bins=n_score_bins,
                **({"label_dimensions": label_dimensions} if label_dimensions else {}),
            )
        except Exception as e:  # noqa: BLE001 - reported together at the end and re-raised
            get_logger(name="main").error(f"{dataset.name}: cube failed: {e}")
            failures.append((dataset.name, e))
            continue
        cube.write(cube_path(work_dir, dataset.name))
        cubes[dataset.name] = cube
        get_logger(name="main").info(
            f"{dataset.name}: cube has {cube.table.num_rows:,} cells over {cube.n_documents:,} documents "
            f"({cube.n_tokens / 1e9:.2f}B estimated tokens); dimensions {cube.dimensions}"
        )

    if failures:
        get_logger(name="main").error(
            f"built {len(cubes)} cube(s); {len(failures)} dataset(s) failed: " + ", ".join(name for name, _ in failures)
        )
        raise failures[0][1]
    return cubes


def load_cubes(work_dir: Path, names: Optional[list[str]] = None) -> dict[str, Cube]:
    """Loads previously built cubes.

    Args:
        work_dir (Path): Working directory holding ``cube/``.
        names (Optional[list[str]]): Restrict to these dataset names.

    Returns:
        dict[str, Cube]: Cubes by dataset name; datasets without a cube are absent.
    """
    cubes: dict[str, Cube] = {}
    directory = Path(work_dir) / "cube"
    if not directory.is_dir():
        return cubes
    for path in sorted(directory.glob("*.parquet")):
        if names and path.stem not in names:
            continue
        cubes[path.stem] = Cube.read(path)
    return cubes


def preview_selection(
    selection_path: Path,
    work_dir: Path,
    force_exact: bool = False,
    allow_sidecar_fallback: bool = False,
    explain: bool = False,
) -> tuple[BlendResult, str]:
    """Costs a selection in documents and tokens.

    Args:
        selection_path (Path): The selection YAML.
        work_dir (Path): Working directory holding the cubes and sidecars.
        force_exact (bool): Scan the per-document sidecars instead of the cubes.
        allow_sidecar_fallback (bool): Permit a sidecar scan for datasets whose cube cannot
            answer a predicate, instead of reporting them.

    Returns:
        tuple[BlendResult, str]: The evaluated blend and its rendered table.
    """
    config = SelectionConfig.from_yaml(selection_path)
    names = [d.name for d in config.enabled_datasets()]
    cubes = load_cubes(work_dir, names)
    sidecars = {name: sidecar_dir(work_dir, name) for name in names}
    result = evaluate_blend(
        config,
        cubes,
        sidecar_dirs=sidecars,
        force_exact=force_exact,
        allow_sidecar_fallback=allow_sidecar_fallback,
    )
    report = format_blend_report(result, datasets_in_order=names, config=config)

    if explain:
        # Attribution needs the cubes, so it is only available on the cube path; an exact
        # sidecar scan does not produce per-cell weights to slice.
        sections = ["", "per-predicate attribution"]
        for dataset in config.enabled_datasets():
            cube = cubes.get(dataset.name)
            if cube is None or not dataset.predicates:
                continue
            try:
                sections.append(predicate_breakdown(cube, dataset, config.policy_for(dataset)).describe())
            except SelectionError as e:
                sections.append(f"  {dataset.name}: {e}")
        report = report + "\n".join(sections)

    return result, report


def apply_selection(
    selection_path: Path,
    registry_path: Path,
    work_dir: Path,
    output_dir: Path,
    show_progress: bool = True,
    allow_overexposure: bool = False,
) -> Path:
    """Writes a selection out as filtered index files plus a manifest.

    Args:
        selection_path (Path): The selection YAML.
        registry_path (Path): The corpus registry YAML.
        work_dir (Path): Working directory holding the sidecars.
        output_dir (Path): Directory receiving the index trees and manifest.
        show_progress (bool): Whether to show progress bars.
        allow_overexposure (bool): Proceed even when the run would repeat data past a
            declared cap.

    Returns:
        Path: Path to the written manifest.
    """
    config = SelectionConfig.from_yaml(selection_path)
    registry = CorpusRegistry.from_yaml(registry_path)
    return materialize_blend(
        config=config,
        registry=registry,
        sidecar_root=Path(work_dir) / "sidecar",
        output_root=output_dir,
        show_progress=show_progress,
        allow_overexposure=allow_overexposure,
    )


def write_packing_configs(
    manifest_path: Path,
    registry_path: Path,
    template_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Renders one packing config per source file of a materialised selection.

    The written configs point ``pack_encoded_data`` at a filtered index, so packing
    tokenizes only the selected documents. Everything else -- tokenizer, jq pattern,
    queue sizes -- is copied from the template.

    Args:
        manifest_path (Path): The ``mix_manifest.yaml`` written by the apply stage.
        registry_path (Path): The corpus registry YAML.
        template_path (Path): A packing config to use as the template.
        output_dir (Path): Directory receiving the rendered configs.

    Returns:
        list[Path]: The written config paths.
    """
    with Path(manifest_path).open() as f:
        manifest = yaml.safe_load(f)
    with Path(template_path).open() as f:
        template = yaml.safe_load(f)
    registry = CorpusRegistry.from_yaml(registry_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dataset in manifest["datasets"]:
        # Bucket rows are named "<dataset>__<level>", so the registry lookup uses the source.
        entry = registry.get(dataset.get("source_dataset") or dataset["name"])
        for source_path, index_path in dataset["index_files"].items():
            relative = Path(source_path).relative_to(entry.jsonl_root)
            config = dict(template)
            config["settings"] = {
                **template.get("settings", {}),
                "src_path": source_path,
                "index_path": index_path,
                "dst_path": str(output_dir / dataset["name"] / relative.with_suffix(".pbin")),
            }
            config_path = output_dir / dataset["name"] / relative.with_suffix(".yaml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            written.append(config_path)
    return written
