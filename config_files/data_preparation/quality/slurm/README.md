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

## Steps 1-4: once per blend

Nothing is written into `/data/annealing`. Indexes go to `$WORK/idx`.

```bash
# 1. Token calibration. Minutes. One task.
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
