# Quality-based selection and up/downsampling

Builds a training blend by filtering documents on quality signals and choosing how
heavily each dataset is sampled. Two kinds of signal are addressed with the same syntax:
metrics a corpus already carries in its records, and external per-document annotations
that are joined on.

The source data is never copied or modified. Selection produces a filtered `.idx`, and
`pack_encoded_data` tokenizes exactly the documents its index lists, so an ablation costs
megabytes of index rather than a second copy of the corpus.

## Two files describe a blend

`annealing_registry.yaml` — where each dataset lives, how it joins to annotations, and
which native metrics to read out of it. Written once and changed rarely.

`annealing_selection.yaml` — thresholds and an up/downsample ratio per dataset. This is
the file you edit per ablation.

## Workflow

```bash
REG=config_files/data_preparation/quality/annealing_registry.yaml
SEL=config_files/data_preparation/quality/annealing_selection.yaml
WORK=/path/to/scratch/blend_v1

# 1. Measure how each corpus's records map to our tokenizer's counts.
#    Reuses a packing config so the calibration tokenizer cannot drift from the
#    tokenizer the packing will actually use.
modalities quality calibrate --registry $REG --work_dir $WORK \
    --tokenizer_config config_files/data_preparation/packed_cc_en_2048.yaml

# 2. One row per document: position, length, estimated tokens, join key, native metrics.
#    The only stage that reads the raw data, so run it as an array. Work is divided by
#    file across all datasets, so one array covers the whole blend.
modalities quality build-sidecar --registry $REG --work_dir $WORK --index_root $WORK/idx \
    --shard_id $SLURM_ARRAY_TASK_ID --num_shards 64

# 3a. Partition the annotation splits by a hash of their key. The expensive half of
#     the join -- HPLT alone is ~12 bn rows -- so run it as an array.
modalities quality bucket-annotations --registry $REG --work_dir $WORK --num_buckets 1024 \
    --shard_id $SLURM_ARRAY_TASK_ID --num_shards 64

# 3b. Attach the labels to each dataset's sidecar and report coverage. Cheap; one task.
modalities quality join-annotations --registry $REG --work_dir $WORK

# 4. Aggregate, so any threshold combination can be costed without reading the sidecars.
modalities quality build-cube --registry $REG --work_dir $WORK

# 5. Edit $SEL and re-run this as often as you like. It reads only the cubes.
modalities quality preview --selection $SEL --work_dir $WORK

# 6. Write the filtered indexes and a manifest recording what was selected.
modalities quality apply --selection $SEL --registry $REG --work_dir $WORK --output_dir $WORK/blend

# 7. Render one packing config per source file, each pointing at its filtered index.
modalities quality write-packing-configs --manifest $WORK/blend/mix_manifest.yaml \
    --registry $REG --template config_files/data_preparation/packed_cc_en_2048.yaml \
    --output_dir $WORK/packcfg

# 8. Pack. Only the selected documents are tokenized.
modalities data pack_encoded_data $WORK/packcfg/<dataset>/<shard>.yaml
```

Steps 1–4 are run once per blend. Step 5 is the loop you actually iterate in.

## What it costs

Measured on `/data/annealing` (43 TB across 19 datasets, ~7.6 bn documents):

| Stage | Cost | How often |
|---|---|---|
| calibrate | minutes | once per blend |
| build-sidecar | ~7 h on 64 tasks (first run; ~3.5 h after) | once per blend |
| bucket + join annotations | ~0.5 h on 64 tasks | once per blend |
| build-cube | ~50 min, single task | once per blend |
| **preview** | **~10 s for the whole blend** | **every threshold you try** |
| apply | ~1 h | once you have settled |
| pack | proportional to what survived | once you have settled |

Changing thresholds, ratios or the missing-annotation policy costs only a `preview`.
Adding a dataset or a native metric means rebuilding that dataset's sidecar and cube,
because the metric has to come out of the raw records.

Two things dominate if you get them wrong, both measured:

* **Keep native-metric patterns to plain field paths** (`.fw_edu_scores`,
  `.metadata.dclm_plus2."__label__1"`). Those are evaluated by direct dictionary lookup.
  Anything jq cannot reduce to a field chain -- pipes, filters, indexing -- falls back to
  jq, which re-serialises the whole document per call and costs about 13x more for the
  entire pass. `build-sidecar` warns when a pattern takes that route.
* **Sequential read from `/data` runs at ~282 MB/s per stream and ~3.8 GB/s aggregate.**
  With plain paths the sidecar pass reaches ~374 MB/s per core, so it is I/O-bound and
  more than ~16 concurrent tasks buys little.
* **`build-sidecar` reads every file twice on its first run**: once for `IndexGenerator`
  to write the `.idx`, then once to build the sidecar. The floor is therefore
  `2 x 43 TB / 3.8 GB/s`, about 6.3 h -- not the 3.2 h a single pass suggests. Measured
  The real run took ~15 h rather than the ~7 h that floor implies, for a reason that is
  about placement rather than throughput -- see the next point.

  The index pass is not throwaway work: `pack_encoded_data` needs those `.idx` files and
  every later run reuses them, so a second `build-sidecar` over the same data -- after
  adding a native metric, say -- is roughly twice as fast.
* **Per-node bandwidth binds before cluster aggregate does, so spread the tasks.** The
  3.8 GB/s above was measured on the login node and does not transfer to a compute node.
  On the real run SLURM packed 12 of the 64 tasks onto one node and 8 onto another while
  four nodes sat idle; each got ~95 MB/s against the 282 MB/s a single stream gets, and
  the job spent a nine-hour tail at a third of its starting speed.

  Both array scripts therefore request 8 CPUs per task. A task is single-threaded and
  needs one; the request exists to cap how many bandwidth-hungry tasks land on a node
  (4 rather than 16), spreading an array of 64 across the cluster. Raise the array size
  only after the tasks are spread -- more tasks on the same node buys nothing.

## What the preview reports

```
dataset        docs kept    row%   tokens kept    tok%   ratio     effective   share
------------------------------------------------------------------------------------
hplt-de           1.42B    61.3%       238.10B   58.4%    0.60       142.86B   18.1%
finepdfs-en       0.31B    44.0%        97.40B   51.2%    1.40       136.36B   17.3%
------------------------------------------------------------------------------------
TOTAL                                                                789.20B  100.0%

target 400.00B tokens -- 389.20B over (97.3%)
```

`row%` and `tok%` differ on purpose. Quality correlates with length, so a filter that
keeps the better documents keeps a larger share of the tokens than of the documents —
which is why the row retention alone cannot be used to predict a token budget.

A `~` next to a row means a numeric threshold fell inside a cube bin rather than on its
edge, so that row was interpolated. Re-run with `--exact` to scan the per-document
sidecars instead.

## Applying the ratio at training time

The ratio is not baked into the data. Use the `weighted_combined` dataset and read the
per-dataset ratios out of `mix_manifest.yaml`:

```yaml
train_dataset:
  component_key: dataset
  variant_key: weighted_combined
  config:
    seed: 42
    repeat_factors: [0.6, 1.4, 2.0]     # from mix_manifest.yaml
    datasets:
      - component_key: dataset
        variant_key: packed_mem_map_dataset_continuous
        config:
          raw_data_path: /path/to/hplt-de.pbin
          sequence_length: ${settings.step_profile.sequence_length}
          sample_key: ${settings.referencing_keys.sample_key}
      # ... one entry per dataset, in the same order as repeat_factors
```

A factor of 2.0 draws a dataset twice per epoch, 0.6 draws six tenths of it. Nothing is
duplicated on disk, and changing the blend means changing a number rather than rebuilding
data.

## Two things to be careful about

**Token counts are estimates.** They are measured per document from the text, using a
per-dataset bytes-per-token ratio or a rescaled native token count. On a synthetic
end-to-end check the estimate came within 0.03% of the packed total, but validate it on
your own data by comparing the preview against the packed result for one small dataset
before trusting a large budget.

**Decide what to do with unannotated documents.** `missing_annotation: keep` treats an
annotation predicate as satisfied for documents that have no label; `drop` treats it as
failed. On a partly downloaded annotation split the unannotated documents can be the
majority, and the two policies then give completely different blends. The join report
(`join_report.json` in the working directory) tells you the coverage per dataset.
