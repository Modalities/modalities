# Quality-based selection and up/downsampling

Builds a training blend by filtering documents on quality signals and choosing how
heavily each dataset is sampled. Two kinds of signal are addressed with the same syntax:
metrics a corpus already carries in its records, and external per-document annotations
that are joined on.

The source data is never copied or modified. Selection produces a filtered `.idx` naming
exactly the documents that survived, so trying a different threshold costs megabytes of index
rather than a second copy of the corpus. Only the final export writes document bytes.

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

# 7. Export the selected documents as JSONL, with the sampling baked into the bytes.
modalities quality export-jsonl --manifest $WORK/blend/mix_manifest.yaml \
    --registry $REG --output_dir $WORK/out
```

The training set is `cat $WORK/out/*/*.jsonl`. Tokenization is left to whatever consumes it.

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
| export-jsonl | proportional to what survived | once you have settled |

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

  The index pass is not throwaway work: the export reads those `.idx` files and every later
  run reuses them, so a second `build-sidecar` over the same data -- after
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

**How wrong can the interpolation be?** Measured on the smoke blend, comparing the cube
against the exact per-document figures `apply` writes into the manifest:

| dataset | predicate kind | cube tokens | exact tokens | error |
|---|---|---|---|---|
| finewiki-de | ordinal only | 18.51M | 18.51M | 0.00% |
| climbmix-en | ordinal only | 54.53M | 54.53M | 0.00% |
| klettermix-de | ordinal + score | 19.73M | 19.80M | -0.35% |
| dolmino | score only | 46.58M | 48.32M | -3.60% |
| finepdfs-es | ordinal + score | 9.73M | 8.64M | +12.61% |
| **total** | | **150.80M** | **149.80M** | **+0.67%** |

Ordinal predicates are exact -- levels are cube dimensions, so no interpolation happens.
Numeric thresholds are only exact when they land on a bin edge, and the error does not
shrink with dataset size, because the bin count is fixed rather than proportional. Use the
cube to explore, then `--exact` before committing to a budget, or raise
`--num_score_bins` when building the cube. `apply` always scans the sidecar, so the
manifest it writes carries exact figures regardless.

### When a predicate is not in the cube

The join attaches twelve annotation columns; the cube groups on seven (`audience_level`,
`commercial_bias`, `content_length`, `content_ratio` and `time_sensitivity` are attached but
not grouped), because grouping on all twelve would multiply the cell count by thousands. A
native metric that is null throughout a dataset is dropped from its cube too.

Thresholding a field the cube does not carry is therefore expressible in a selection but
cannot be answered from the cube, and `preview` refuses rather than quietly reading every
document:

```
these predicates cannot be answered from the cubes:
  nemotron-cc: cube for 'nemotron-cc' was not grouped on label 'commercial_bias' ...
    (answering it from the sidecar means reading 1,696,565,570 documents)
```

Three ways out: drop or replace the predicate; rebuild that dataset's cube naming the field
(`build-cube --only nemotron-cc --label_dimension commercial_bias --label_dimension ...`,
listing the others too, since the flag replaces the default set); or accept the cost with
`--allow_fallback`. That last is what used to happen silently, and it turned a 13-second
preview into a job still running after ten minutes.

## The ratio is applied by the export, not at training time

`export-jsonl` materialises the sampling: a dataset at 2.0 has each of its documents written
twice, one at 0.6 has two of every five dropped. The training set is the concatenation of the
exported files, drawn once.

This is a change from how the pipeline used to work, and the trap is worth stating plainly.
`mix_manifest.yaml` still records `ratio: 2.0`, because that is what was asked for. Feeding
that number to a `weighted_combined` dataset now would apply it a **second** time. Read
`export_manifest.yaml` instead: it reports `training_ratio: 1.0` and
`repeat_factor_applied: true`.

Fractional factors are resolved per document rather than by truncating a list -- 1.2 means
every document once and a hash-chosen fifth of them twice -- keyed on the selection's `seed`
and the document's position, so it is reproducible across runs and machines. Copies of a
document are written adjacent to one another, so a sequential reader needs a shuffle buffer
larger than the run of copies to separate them. Nothing is
duplicated on disk, and changing the blend means changing a number rather than rebuilding
data.

## Quality-aware upsampling curves

A `ratio` treats every surviving document alike: `nemotron-cc` filtered to "content quality
at least adequate" and set to 1.2 repeats the barely-adequate documents exactly as often as
the excellent ones. A curve makes the repeat factor rise with quality instead, so the token
budget is spent where the quality signal says it should be.

```yaml
- name: nemotron-cc
  upsampling:
    quality_field: content_quality      # ordinal axis, worst to best
    target_tokens: 900_000_000_000      # or target_ratio, as a multiple of what is available
    max_factor: 7.0                     # no bucket repeated more than this
    discard_below_percentile: 40        # drop the weakest 40% of tokens
  predicates: [...]                     # still applied first; the curve works on survivors
```

`ratio` and `upsampling` are mutually exclusive -- both express the same decision, so
setting both is refused rather than one being silently ignored.

The method and the functional form come from Dolma 3 / Olmo 3 (arXiv:2512.13961 §3.4.4,
appendix A.2.4), which measured it against flat quality filtering on 1B models and found it
better at every matched repetition factor -- 0.740 against 0.843-0.870 bits-per-byte on their
maths suite. Quality goes on a [0, 1] axis where each bucket's width is its share of the
dataset's tokens, and the repeat factor is `C * (x - a)**p` above the discard threshold `a`,
with `p` chosen so the top bucket sits exactly at `max_factor`.

Reproducing their published example -- twenty equal vigintiles, discard the bottom 40%,
repeat at most 7x, draw as many tokens as the pool holds -- gives exactly their figure: the
bottom eight buckets dropped, the top at 7.00x, monotone in between.

**What `apply` does with it.** Documents with different factors have to live in different
indexes, so each bucket becomes its own index tree and its own manifest row, named
`<dataset>__<level>` with `source_dataset` naming the registry entry. The curve is re-solved
from the exact token counts found during `apply` rather than from the cube, since that stage
reads every document anyway. `export-jsonl` then merges the buckets back into one output
directory per input dataset, each bucket repeated by its own factor, with the documents
written in source order rather than grouped by quality level.

### Two limits worth knowing

**The axis is only as fine as the field.** Dolma 3 cuts twenty vigintiles because their
quality signal is a continuous classifier score. Propella labels are ordinal with about five
levels, and the levels are not evenly filled: on the smoke blend, `climbmix-en` has 81.9% of
its tokens in `moderate` and 17.9% in `high`. A curve over that axis can still do something
useful -- it drew 2.53x the `high` tokens and 6% of the `moderate` ones -- but it cannot
express a finer preference than the levels allow. A threshold falling inside a level applies
a fractional factor to the whole level rather than splitting it, and which documents survive
is then whatever the dataloader's even spread picks.

**Only ordinal fields for now.** A native numeric metric would need the cube's quantile edges
carried into materialisation, which is not built. Setting a non-ordinal `quality_field` is
refused when the config loads, rather than after a preview has already succeeded.

## Repetition is per pass, so declare what the run consumes

A `ratio` of 2.0 draws a dataset twice *per pass over the blend*. If the run consumes more
tokens than the blend yields, the loader comes round again and every factor is multiplied by
the number of passes. A bucket set to 7x under a curve becomes 14x on a second pass, which is
well past the point where repetition pays for itself.

`target_tokens` is what the run will actually consume, and it is what makes this checkable:

```yaml
target_tokens: 400_000_000_000     # what the run consumes
max_total_exposure: 3.0            # refuse if anything is seen more often than this
```

A dataset with an upsampling curve uses its own `max_factor` as the cap instead, since that is
already declared. `preview` always reports the accounting; `apply` refuses, because that is the
point of commitment:

```
repetition, once wrapping is counted
  run consumes 300,000,000 tokens from 150,800,110 effective -- wraps 1.99 passes
                             factor  exposure     cap
    climbmix-en / high         2.53      5.04       4  OVER
    klettermix-de              2.00      3.98     2.5  OVER
```

Pass `--allow_overexposure` to `apply` to proceed anyway. Without `target_tokens` the caps
still apply to the requested factors, but the multiplier from wrapping is unknown and the
report says so rather than assuming one pass.

## Which predicate is actually doing the work

`preview --explain` attributes a dataset's retention to its individual predicates and shows
how they overlap. It costs milliseconds, because the per-cell weights are the whole
computation and the cube is already loaded.

```
  finepdfs-es: 9,727,716 of 65,540,305 tokens kept
    predicate                                         matches   share       marginal
    fw_edu gte 1.5                                  9,748,842  14.9%     55,753,485 ~
    content_integrity at_least mostly_complete     65,481,201  99.9%         21,126
```

`marginal` is the number to read: how many more tokens the dataset would keep if that
predicate were dropped and the others left alone. A marginal of zero means the predicate is
fully shadowed by its neighbours -- it changes nothing and only makes the selection harder to
read. On the smoke blend this immediately found two: `content_integrity` on `finepdfs-es`
(21 k marginal out of 65 M) and `educational_value at_least basic` on `klettermix-de`, which
matches 100.0% of tokens.

The overlap matrix below the table gives tokens matching each pair. Its diagonal is each
predicate's own match count -- not the product with itself, which for an interpolated
predicate squares a fractional survival rate and undercounts.

## The source tree must not move underneath a sidecar

A sidecar row locates its document by `(file_id, byte_offset, byte_len)`, where `file_id`
is a position in the dataset's sorted file list. If the source tree changes between
building the sidecar and using it, the same id names a different file and every offset is
wrong.

This is not hypothetical. A transfer re-sharded four corpora after their sidecars were
built -- `Nemotron-CC` went from 606 MB files to 137 MB files while keeping nearly the
same file *count* -- and the only check that existed compared counts. Eleven of nineteen
datasets had unusable sidecars, and just one of them failed loudly.

So `build-sidecar` records the file list it used in `sidecar/<dataset>/_files.json`, and
`apply` resolves ids through that list and refuses to run if any recorded file has
changed size or vanished. Paths are stored relative to the root, so moving or snapshotting
a tree is fine; only the file set has to agree.

Run the direct check after any transfer, and before `apply` on a tree that might have been
touched:

```bash
$MQ -m modalities quality verify-sidecar --registry $REG --work_dir $WORK
```

It seeks to recorded offsets and compares the document it finds against the recorded text
length, so it catches a file that was rewritten at the same size. It skips rows at offset
zero on purpose: the first document of any JSONL file parses, so those rows succeed even
against a completely different file, and sampling them is exactly how a broken sidecar
looked healthy. `--adopt` stamps a manifest onto a sidecar built before manifests existed,
but only if it verifies.

A drifted dataset needs its sidecar rebuilt, then re-joined and re-cubed. The annotation
buckets are unaffected -- they are built from the annotation cache, not the corpora -- so
the expensive bucketing stage does not repeat.

## Testing the pipeline end to end

`slurm/make_smoke_snapshot.py` freezes about 1 GB of five corpora into a snapshot, and
`smoke_registry.yaml` / `smoke_selection.yaml` run the whole pipeline over it in minutes.
The five datasets cover all four distinct join-key kinds plus the native-metrics-only
path, which is every branch of the join; HPLT is left out because it shares FineWiki's key
kind and would add 327 GB of bucket reads to exercise no new code.
`slurm/check_token_estimates.py` then tokenizes a sample of the exported JSONL and compares
it against the preview's estimates, and `slurm/verify_jsonl.py` replays sampled shards against
the corpus and asserts nothing was written into the source tree.

Use it after any change to the sidecar, join, cube, or materialize stages. It is much
cheaper than discovering a bug 15 hours into a real build.

## Two things to be careful about

**Token counts are estimates, and a corpus has no single bytes-per-token ratio.** On the
German FineWiki snapshot the ratio runs from 3.571 for documents under a kilobyte to 34.648
for the nine documents above 256 KB -- and those nine hold 5.7% of all bytes. So the
calibration measures a ratio per log-spaced size stratum and applies it per document from
the byte length the sidecar records exactly. Getting there needed the sample to be drawn at
offsets spread across each file, taking the document *containing* each offset so that
selection is proportional to length; sampling uniformly by document count found 2 of the
top-stratum documents per 2,000 draws, where length-proportional sampling finds 44.

Measured against the true token count of all 27,846 documents, worst case over five seeds:
16.2% error for the original estimator (a fixed bias, identical across seeds, so it looked
stable), 19.4% for a global ratio on a spread sample, 0.8% for the stratified estimate.

Datasets carrying their own token count in every record (FinePDFs, KletterMix, FinePhrase)
are estimated from that field rescaled to our tokenizer, per document, and need no
stratifying.

Still validate on your own data: export one small dataset and run
`slurm/check_token_estimates.py`, which tokenizes a sample and reports the error per dataset.
Nothing in the pipeline tokenizes any more, so this is the only check on the model the whole
token budget rests on. Document counts, by contrast, are not estimates -- the filtered index
names precisely the selected documents -- and `verify_jsonl.py` requires them to match
exactly.

**Decide what to do with unannotated documents.** `missing_annotation: keep` treats an
annotation predicate as satisfied for documents that have no label; `drop` treats it as
failed. On a partly downloaded annotation split the unannotated documents can be the
majority, and the two policies then give completely different blends. The join report
(`join_report.json` in the working directory) tells you the coverage per dataset.
