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
export MQ=/data/user/richard.rutmann/venvs/modalities-quality/bin/python
export REPO=/home/richard.rutmann/repos/modalities
export QDIR=$REPO/config_files/data_preparation/quality
export WORK=/data/user/richard.rutmann/annealing_blend
```

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

# 4. Join, then aggregate. One task each; needs 2 and 3 finished.
sbatch $QDIR/slurm/3_join_and_cube.sbatch
```

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
