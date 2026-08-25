# Running the quality pipeline on /data/annealing

`/home/richard.rutmann/data` is a symlink to `/data`, so `/home/richard.rutmann/data/annealing`
and `/data/annealing` are the same tree (same inode). The registry already uses the
`/data/...` form; nothing needs changing.

## Environment

The `nemo_25_11.sif` container cannot run this branch: it ships torch 2.9 and modalities
needs `>=2.10` (`ScheduleDualPipeV`). These stages are CPU-only, so a plain venv is
simpler than rebuilding the container:

```bash
/opt/conda/bin/python3 -m venv /data/user/richard.rutmann/venvs/modalities-quality
V=/data/user/richard.rutmann/venvs/modalities-quality
$V/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$V/bin/pip install -e /home/richard.rutmann/repos/modalities
```

Already built and verified at that path. Every command below uses `$MQ`:

```bash
cd /home/richard.rutmann/repos/modalities
source config_files/data_preparation/quality/slurm/env.sh
```

`env.sh` sets `MQ`, `REPO`, `QDIR`, `WORK`, `HF_HOME`, `HF_TOKEN` and `EXPORTS`, only
filling in what is unset, and refuses to continue if any of them ends up empty or points
at something missing. Source it in every new shell. Setting them by hand works too, but a
shell where `QDIR` is empty turns `$QDIR/slurm/x.sbatch` into `/slurm/x.sbatch`, and one
where `WORK` is empty turns `rm -rf $WORK/buckets` into `rm -rf /buckets`. Both have
happened.

`calibrate` downloads the tokenizer, so it needs HF auth. A token is already at
`~/.config/huggingface/token`:

```bash
export HF_TOKEN="$(tr -d '\r\n' < ~/.config/huggingface/token)"
export HF_HOME=/data/cache/hf_cache
```

## Before you start

Confirm the tokenizer in `annealing_packing_template.yaml`. It is set to
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, the one the long-context pipeline uses;
the token audit under `/data/michael.fromm` used the Super-120B variant. Every token
figure downstream depends on this, and a wrong choice fails silently.

Two registry entries are deliberately disabled or worth a second look:
`finepdfs-it` is disabled (it is English data duplicating `finepdfs-en`), and
`nemotron-cc-v2` is enabled at 9.86 TB but has no annotations at all. Drop it from the
registry if the blend does not need it -- it is ~16 h of sidecar work.

## Running everything with timings

`run_all_timed.sh` runs the steps in order and prints how long each took, plus a summary.
Array jobs go through `sbatch --wait`, so the timing is the job's real duration rather
than how long submission took.

```bash
bash $QDIR/slurm/run_all_timed.sh            # all steps
bash $QDIR/slurm/run_all_timed.sh 1 2 3 4    # only the once-per-blend stages
bash $QDIR/slurm/run_all_timed.sh 5          # just re-preview
```

It stops at the first failing step and still prints the summary, and it honours
`REGISTRY`, `SELECTION`, `TOKENIZER_CONFIG`, `WORK`, `SIDECAR_TASKS`, `BUCKET_TASKS`,
`NUM_BUCKETS`, `PACK_TASKS`, `SAMPLE_SIZE` and `BLEND_NAME` as environment overrides.

The individual commands are below if you would rather drive them by hand.

## Steps 1-4: once per blend

Nothing is written into `/data/annealing`. Indexes go to `$WORK/idx`.

```bash
# 1. Token calibration. ~4 min for all 19 datasets, one task, ~1 GB read.
#    Writes calibration.yaml after each dataset, so an interruption keeps what it
#    measured; re-run with --only to fill in the rest.
$MQ -m modalities quality calibrate \
    --registry $QDIR/annealing_registry.yaml --work_dir $WORK \
    --tokenizer_config $QDIR/annealing_packing_template.yaml --sample_size 2000

# 2. Sidecar. The only stage that reads the raw data. Array of 64.
sbatch $QDIR/slurm/1_build_sidecar.sbatch

# 3. Bucket the annotation splits. Array of 64. Wait for step 2 only if you
#    like -- it touches different data, so the two can run concurrently.
sbatch $QDIR/slurm/2_bucket_annotations.sbatch

# 4a. Join, one array task per annotated dataset. The joins are independent, so wall
#     time is the slowest dataset (~2 h for nemotron-cc) not the sum (~6 h).
sbatch --export=$EXPORTS --array=0-15 $QDIR/slurm/3a_join_annotations.sbatch

# 4b. Aggregate into cubes. After every join task has finished. ~50 min, one task.
sbatch --export=$EXPORTS $QDIR/slurm/3b_build_cubes.sbatch
```

`3_join_and_cube.sbatch` still exists and does both in one task if you would rather not
manage two submissions; it takes ~6 h instead of ~3 h.

Check coverage before trusting anything: `$WORK/join_report.json` gives the annotated
fraction per dataset. FinePDFs was 10-34% at last measurement.

## Step 5: the loop you actually iterate

```bash
# Edit thresholds and ratios, then:
$MQ -m modalities quality preview \
    --selection $QDIR/annealing_selection.yaml --work_dir $WORK
```

~10 s for the whole blend, on the login node. Reads only the cubes. Repeat as often as
you like. `--exact` scans the per-document sidecars instead, for a threshold that fell
inside a cube bin.

## Steps 6-8: once you have settled

```bash
# 6. Filtered indexes plus a manifest of what was selected.
$MQ -m modalities quality apply \
    --selection $QDIR/annealing_selection.yaml \
    --registry $QDIR/annealing_registry.yaml \
    --work_dir $WORK --output_dir $WORK/blend_v1

# 7. One packing config per source file, each pointing at its filtered index.
$MQ -m modalities quality write-packing-configs \
    --manifest $WORK/blend_v1/mix_manifest.yaml \
    --registry $QDIR/annealing_registry.yaml \
    --template $QDIR/annealing_packing_template.yaml \
    --output_dir $WORK/packcfg

# 8. Tokenize only the selected documents. Array over the generated configs.
find $WORK/packcfg -name '*.yaml' | sort > $WORK/packcfg_list.txt
sbatch $QDIR/slurm/4_pack.sbatch
```

Both steps replace rather than merge, so rerunning them with a changed selection is safe:

* `apply` builds the whole blend in a sibling directory and moves it into place only once
  the exposure check has passed and the manifest is written. A rejected apply leaves the
  previously published blend exactly as it was. It needs room for two copies of the index
  trees while the move happens -- 19 GB each on the current annealing blend.
* `write-packing-configs` deletes the `.yaml` and `.pbin` files the new manifest no longer
  names, and logs every removal. This matters because step 8 globs the directory for jobs
  and the loader globs it for `.pbin` files: a config left over from a wider selection
  would otherwise be packed and trained on. Pass `--no_prune` to keep them, e.g. when two
  selections deliberately share one packing directory.
* It also writes a `.fingerprint` beside each config, covering the source file's size and
  mtime, the contents of the filtered index, the tokenizer and the jq/eod settings. This
  catches the case a name check cannot: changing a predicate rewrites an index *at the
  same path*, so the `.pbin` next to it keeps its name while holding the previous
  selection's documents, and step 8 skips it as already done. `pack_many.py` records the
  fingerprint it packed from and skips only an output whose record still matches.
* `pack_many.py` packs into `<name>.pbin.partial` and moves it into place once finished,
  tearing up any existing fingerprint record before it starts. A killed job therefore
  leaves a `.partial` and no record, rather than a half-written `.pbin` that the previous
  record still vouches for. Regenerating the configs clears stray `.partial` files.

The blend packed before fingerprinting has no records, so the next config regeneration
would treat all 54,738 outputs as stale and repack them -- about 1 h 40 m and 6 TB of
rewriting. If the packed data really does come from the current manifest, adopt it once
instead:

```bash
$MQ -m modalities quality write-packing-configs --manifest $WORK/mix/mix_manifest.yaml \
    --registry $REG --template $TPL --output_dir $WORK/packcfg --adopt_existing
```

Only for that migration. After changing a selection it would assert something false and
keep exactly the stale outputs the fingerprint exists to catch.

Then take the `ratio` values out of `mix_manifest.yaml` into a `weighted_combined`
dataset in the training config, as shown in the parent README.

## Validate the token estimate before trusting a large budget

Predicted token counts are estimates. Confirm them on one small dataset by comparing the
preview against what packing actually produced:

```bash
$MQ -m modalities quality preview --selection $QDIR/annealing_selection.yaml \
    --work_dir $WORK 2>&1 | grep finewiki-it
$MQ -c "
from pathlib import Path
from modalities.dataloader.dataset import PackedMemMapDatasetBase
total = 0
for p in Path('$WORK/packcfg/finewiki-it').rglob('*.pbin'):
    d = PackedMemMapDatasetBase(p, sample_key='text', load_index=True)
    total += sum(len(d[i]['text']) for i in range(len(d)))
print('actual tokens:', total)
"
```

On a synthetic end-to-end check the estimate was within 0.03%. Measure it here before
scaling the conclusion to 43 TB.

## Resuming an interrupted join

The join writes labels into each sidecar part as it goes, so an interrupted run can be
continued rather than redone:

```bash
JOIN_RESUME=1 sbatch --wait --export=$EXPORTS,JOIN_RESUME=1 \
    --array=<task id> $QDIR/slurm/3a_join_annotations.sbatch
```

`--resume` skips parts that already carry the label columns and reports how many it
skipped. A real interrupted `nemotron-cc` join finished in **9 minutes instead of 12
hours** this way, having skipped 5,309 of 5,319 parts.

Leave it off after re-bucketing the annotations: resuming would keep the labels from the
previous bucketing rather than picking up the new ones.

## Clearing the bucketed annotations

A sharded bucketing run cannot clear its own output directory -- sibling tasks are writing
into it -- so clearing it is a separate, deliberate step:

```bash
bash $QDIR/slurm/reset_buckets.sh
```

It refuses to run without `WORK` set, and refuses obviously wrong paths. Sidecars are left
alone; only `$WORK/buckets` goes.

## Re-running a failed shard

Every stage is idempotent per shard, so a failed array task is re-run on its own:

```bash
# sidecar: rebuilds only that task's parquet parts
sbatch --array=17 $QDIR/slurm/1_build_sidecar.sbatch

# bucketing: --num_shards must match the original array, or the buckets will not line up
sbatch --array=41 $QDIR/slurm/2_bucket_annotations.sbatch
```

`bucket-annotations` skips a split whose output is already complete, so re-submitting the
whole array is safe but pointless. Use `--force` to genuinely re-bucket.

## What the stages leave behind

```
$WORK/calibration.yaml          bytes-per-token per dataset
$WORK/idx/<dataset>/            .idx per source file  (reusable; packing needs these too)
$WORK/sidecar/<dataset>/        one parquet part per source file
$WORK/buckets/<split>/          partitioned annotations
$WORK/cube/<dataset>.parquet    what preview reads; a few MB each
$WORK/join_report.json          annotated fraction per dataset  <- read this
$WORK/blend_v1/                 filtered indexes + mix_manifest.yaml
$WORK/packcfg/                  generated packing configs and the resulting .pbin
```

Only `$WORK` is written. `/data/annealing` is read-only throughout -- verified on a real
run of `finewiki-it`, which produced 452,714 sidecar rows and added no file to the source
tree.

## Measured on a real run of finewiki-it

| Stage | Measured |
|---|---|
| calibrate, all 19 datasets | 4 min 5 s (2000 documents sampled each) |
| build-sidecar, 1 of 4 files | 55 s for 6 GB / 452,714 documents, incl. index creation |
| bucket-annotations, finewiki split | 6 m 36 s for 43,097,138 rows (108.8k rows/s) |
| join-annotations | 1 m 47 s, **100 % coverage**, 868,586 duplicate annotation keys (2.0 %) |
| build-cube | 12 s -> 456 cells over 452,714 documents, 1.20 B estimated tokens |
| preview | 8 s, of which ~6 s is interpreter startup |

Two things worth carrying forward from that run:

* **Duplicate annotation keys are normal, not a Nemotron-CC quirk.** finewiki reported
  868,586 of them. The join keeps the first occurrence and reports the count; check it in
  `join_report.json` rather than assuming a clean one-to-one join.
* **`preview` pays ~6 s of import overhead** before it does any work, because the CLI
  imports the component registry and therefore torch. The cube evaluation itself is
  milliseconds to a couple of seconds, so budget roughly 15-20 s for a full-blend preview
  rather than the 10 s the cube maths alone suggests.

On that run token retention (62.6 %) exceeded row retention (51.5 %) on real data, which
is the length correlation the design exists to account for.
## Full run, as measured on the annealing blend

Timings are from the complete run over `/data/annealing` (20.21 TB, 19 datasets, August
2026). Everything long goes through SLURM: the login node has usage limits and killed two
sessions during this run.

```bash
cd /home/richard.rutmann/repos/modalities
source config_files/data_preparation/quality/slurm/env.sh
REG=$QDIR/annealing_registry.yaml
SEL=$QDIR/annealing_selection.yaml
TPL=$QDIR/annealing_packing_template.yaml
```

### 0. Gate

Do not start while the corpus is still being written. A transfer that re-shards a corpus
after its sidecar is built invalidates every byte offset, and the only loud symptom is a
dataset whose file count fell to zero.

```bash
find /data/annealing -name '*.jsonl' -mmin -180 | wc -l          # must be 0
$MQ -c "
from pathlib import Path
from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry
r = CorpusRegistry.from_yaml(Path('$REG'))
[print('MISSING', d.name, d.jsonl_root) for d in r.enabled_datasets() if not d.jsonl_root.exists()]"
```

### 1-9. The pipeline

```bash
# 1. Calibrate.  48 min for 19 datasets.
$MQ -m modalities quality calibrate --registry $REG --work_dir $WORK --tokenizer_config $TPL

# 2. Sidecars.  2 h 09 m, 64 tasks, the only stage that reads all 20 TB.
sbatch --wait --export=$EXPORTS $QDIR/slurm/1_build_sidecar.sbatch

# 3. Verify the offsets before spending anything more.  143 s.
$MQ -m modalities quality verify-sidecar --registry $REG --work_dir $WORK

# 4. Buckets.  4.5 h -- SKIP if $WORK/buckets is intact and the annotations have not moved.
# sbatch --wait --export=$EXPORTS $QDIR/slurm/2_bucket_annotations.sbatch

# 5. Join.  ~6 h; nemotron-cc is the long pole at 6 h alone.  Add JOIN_RESUME=1 to continue
#    an interrupted run -- never after re-bucketing, which would keep stale labels.
sbatch --wait --export=$EXPORTS $QDIR/slurm/3a_join_annotations.sbatch

# 6. Cubes.  12 min.
sbatch --wait --export=$EXPORTS $QDIR/slurm/3b_build_cubes.sbatch

# 7. Preview.  14 s -- edit $SEL and repeat as often as you like.
$MQ -m modalities quality preview --selection $SEL --work_dir $WORK
$MQ -m modalities quality preview --selection $SEL --work_dir $WORK --explain   # which predicate binds
$MQ -m modalities quality preview --selection $SEL --work_dir $WORK --exact     # before committing

# 8. Apply.  42 min, peaks near 160 GB: it holds every kept document's offsets.
sbatch --wait --job-name=q_apply --nodes=1 --cpus-per-task=8 --mem=220G --time=12:00:00 \
  --output=$HOME/logs/quality/apply_%j.out --error=$HOME/logs/quality/apply_%j.err \
  --export=$EXPORTS --wrap="srun $MQ -m modalities quality apply --selection $SEL \
    --registry $REG --work_dir $WORK --output_dir $WORK/mix"

# 9. Packing configs.  12 min, one per source file.
$MQ -m modalities quality write-packing-configs --manifest $WORK/mix/mix_manifest.yaml \
    --registry $REG --template $TPL --output_dir $WORK/packcfg
find $WORK/packcfg -name '*.yaml' | sort > $WORK/packcfg_list.txt

# 10. Pack.  1 h 40 m for 6.0 TB.
sbatch --wait --export=$EXPORTS,TEMPLATE=$TPL $QDIR/slurm/4_pack.sbatch

# 11. Verify.  ~26 min: headers, token and document counts, and the blend load.
sbatch --wait --export=$EXPORTS,SOURCE_ROOT=/data/annealing $QDIR/slurm/5_verify.sbatch
```

About 12 hours end to end, with the join as the long pole.

### What each check catches

`verify-sidecar` reads the source bytes at recorded offsets. A re-sharded corpus once left
11 of 19 datasets with unusable sidecars and only one failed loudly.

`5_verify.sbatch` reads packed file headers rather than counting files. One `.pbin` in 54,738
was 151 MB on disk reporting `data_len=0`; counting files would have shipped that dataset
567 M tokens short.

Document counts in the verification must match **exactly**. They are not estimates -- the
filtered index names precisely the selected documents -- so any difference is a defect in
materialize or the index, not estimator error. Token estimates landed within -0.56% overall.
