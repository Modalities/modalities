# Changelog

| PR               | Type       | Ref. Issue(s) | Breaking Changes |PR Description|                                                                                  
|------------------|------------|---------------|------------------|------------------------------------------------------------------------------------------------|
| [#141](#pr-141-towards-stable-modalities-version)  | Bug Fix    |  [#129](https://github.com/Modalities/modalities/issues/129)         | **Yes**              | Towards stable modalities version                                                               |
| [#154](pr-154-manual-swiglu-implementation)  | Bug Fix    |  [#14](https://github.com/Modalities/modalities/issues/14)         | **Yes**              | Towards stable modalities version                                                               |
|    |   |           |        |                                                                |




## PR #141 Towards stable modalities version

This PR further stabilise the codebase and makes training more robust also w.r.t. loss spikes, which we fixed via scaled weight initialisation and an increased batch size in our experiments.
The PR also fixes all failing tests and adds a simple entrypoint for running cpu, single-gpu and multi-gpu tests. The PR contains multiple sub PRs. 

**General changes:**
* Bug fix: the model evaluation mode is now properly deactivated after evaluation (see PR [#131](https://github.com/Modalities/modalities/pull/131))
* Bug fix: Fixed the implementation of Pre-LN for GPT2 model (see PR [#136](https://github.com/Modalities/modalities/pull/136))
* Enhancement: Further mixed precision strategies; also added one matching MegatronLM's.
* Enhancement: Single, unified entrypoint for running cpu, single-gpu and multi-gpu tests. All tests fixed. (PR [#155](https://github.com/Modalities/modalities/pull/155))
* Enhancement: Previously, we would chunk the dataset into `block_size` long chunks. Each chunk would then be used for training individually. As a result, the last token of a block would be only used as a target but never as an input. We changed this, such that we reuse the last token of a batch as the first one of the subsequent batch. (PR [#158](https://github.com/Modalities/modalities/pull/158))
* Bug: Indexing of the original samples of the dataset pbin files had multiple bugs. The index tuples are now always in bytes and the start of the first sample in the data section starts at byte 0 (before the was a wrong offset) (PR [#164](https://github.com/Modalities/modalities/pull/164))
* Enhancement: Improvements on the current pull request template and addition of several issue templates (bug report, documentation, feature request, blank) (PR [#172](https://github.com/Modalities/modalities/pull/172))
* Components and factories for plain, scaled and scaled_embed initialisation. (PR [#161](https://github.com/Modalities/modalities/pull/161))
* in GPT2 model training configs, the standard deviation `std` can now be set to the string `auto` (in which case it will equal `sqrt(2/(5*hidden_dim))`, see e.g. https://arxiv.org/abs/2312.16903) (PR [#161](https://github.com/Modalities/modalities/pull/161))
* The CoCa model, which previously used a hardcoded, (probably not entirely correct) scaled initialization (see #165), can now only use plain initialization (PR [#161](https://github.com/Modalities/modalities/pull/161))


**Breaking changes:** 
* Enhancement: Logging is now always based on #training steps and #consumed tokens (PR [#137](https://github.com/Modalities/modalities/pull/137))
   This change is a breaking change and the experiment configs need to adapated as shown [here](https://github.com/Modalities/modalities/pull/137/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4). 
* Enhancement: The model parameters are now grouped within the respective model. The optimizer can leverage these groups to e.g., only apply weight decay to non-layer-norm weights. See [here](https://github.com/Modalities/modalities/pull/139/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4) for the necessary config changes. (PR [#139](https://github.com/Modalities/modalities/pull/139))
* Enhancement: We support now different attention implementations (manual, pytorch flash, DAO flash) See [here](https://github.com/Modalities/modalities/pull/138/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4) for the respective config changes. (PR [#138](https://github.com/Modalities/modalities/pull/138))
* Enhancement: replaced `block_size` in `Dataset`, `Model` and `NumberConversion` with `sequence_length` (PR [#158](https://github.com/Modalities/modalities/pull/158))
* Enhancement: `block_size` is now `sequence_length +1` and we should always specify `sequence_length` as a value of power of 2. (PR [#158](https://github.com/Modalities/modalities/pull/158))
* Enhancement: Restricted the codebase to the officially supported python versions 3.10 and 3.11 ((PR [#174](https://github.com/Modalities/modalities/pull/174)))
* All training configs require an additional component for initialization of the raw model (i.e. the model with random weights), as shown [here](https://github.com/Modalities/modalities/blob/7d26675051b918c3a2b98f32f50cb3ca8ef97d6f/config_files/training/config_lorem_ipsum.yaml#L181). (PR [#161](https://github.com/Modalities/modalities/pull/161))

## Checklist before submitting final PR
- [ ] My PR is minimal and addresses one issue / enhancement in isolation
- [ ] I have merged main into this feature branch
- [ ] I have reviewed my own code w.r.t. correct implementation, missing type hints, proper documentation, etc.
- [ ] I have run a sample config for model training
- [ ] I have fixed all failing tests (`python tests/tests.py`)



## PR #154 Manual SwiGLU implementation

This [PR](https://github.com/Modalities/modalities/pull/154) adds a manual SwiGLU implementation. The original one from xops was imcompatible with activation checkpointing (see issue [#14](https://github.com/Modalities/modalities/issues/14)) 

**General changes:**
* replaces xops swiglu imlementation with custom reimplementation

**Breaking changes:** 
* renaming of `fused_swiglu` to `swiglu` in `ActivationType` (see [here](https://github.com/Modalities/modalities/pull/154/commits/90fb3bd06a407333423cffeab486711e26ef8ddf) for the respective config changes)

## PR #236 Remove code related to Mamba

This [PR](https://github.com/Modalities/modalities/pull/236) removes all code related to Mamba. The latest state of main with Mamba can be found in the branch main_with_mamba.

**General changes:**
* Removes Mamba-related code

**Breaking changes:** 
* None
 

## PR #254 Warmstart infrastructure switch

This PR mainly addresses the warmstart of model training, e.g., after GPU crashes.

**General Changes**
* Fixes issue #242 
* Warmstarts with changing infrastructure (e.g.,. different number of GPUs) are now supported.
* Restructures the settings part of the configs to 
* Adds various checks for consistency of model training (e.g., target tokens and number of dataset tokens mismatch)
* Refactors all configs to be runnable again
* Adds an interactive jupyter notebook-based Tutorial on how to use Modalities. (merged from PR #239 )
* Adds a warmstart tutorial
* TrainingReportGenerator that creates a report on the training setup and prints out warnings in case of inconsistencies.
* Activation Checkpointing is now a component
* Added further NumberConversion routines

**Breaking Changes**
* the settings part of the configs have been completely refactored



## PR #261 Dataloader inefficiencies fix and combined dataset feature

This PR addresses issue #258 (inefficiencies in the dataloader) and additionally introduces a combined dataset, where a dataset can now comprise a list of datasets and iterate over them.
As part of fixing the dataloader inefficiencies, we now implement the sample skipping functionality not on the dataloader level  anymore but in an adapted version of the PyTorch `DistributedSampler`. I reran a warm start and the learning is equivalent to a full, non-warmstarted run. 

<img width="1415" alt="Screenshot 2024-09-27 at 10 36 19" src="https://github.com/user-attachments/assets/65dfb1ed-e96b-4f50-a127-bc9d240ddff9">


**General Changes**
* Introduced `ResumableDistributedSampler` which is a copy of the PyTorch `DistributedSampler` added with the feature to skip samples. This is from now on used for warmstarts instead of the `skip_num_samples` in the Dataloader. In case of skipping samples, the dataloader had to instantiate a `ResumableBatchSampler` which was internally iterating over all the dataset indices. For small datasets this was fine, but for larger datasets (in the trillion token range) this became a bottleneck at instantiation time:
https://github.com/Modalities/modalities/blob/b79d04d3e92d0845c5ec91f8dd41176fd543cb23/src/modalities/dataloader/samplers.py#L25-L28
Skipping in the  `ResumableDistributedSampler` is skipping in O(1) now. The `ResumableBatchSampler` was removed from the codebase.
* Replaced the packed index generation routine (inefficient due to for loop)
https://github.com/Modalities/modalities/blob/b79d04d3e92d0845c5ec91f8dd41176fd543cb23/src/modalities/dataloader/dataset.py#L331-L334
with a vectorized version.
* added new `NumberConversion` routine `num_samples_from_num_tokens `

**Breaking Changes**
* Removed RepeatingDataloader, as a feature that was never actively used for running multiple epochs and had complex maintenance when refactoring the sampling. If needed we could reimpliment it. 
*  In the settings, the `training_progress` section has now `num_seen_samples` instead of `local_num_seen_batches `, as skipping is now done on the Sampler level and not on the dataloader level anymore
* `batch_size ` and `fast_forward_batch_id ` fields in the `LLMDataLoader ` are not neede anymore and were removed.


## PR #269 Large file reader efficiency improvements and byte reading support

This PR makes the LargeFileLinesReader about 50% faster by using mmap instead of file seek operations. 
We can also now configure the encoding used for reading the documents. If encoding is specifically set to None (default is utf-8), we return the document as a byte string not enforcing any encoding. This is especially helpful when we e.g., sample from the data and want to create a subset of the dataset. In this case, we can just pass around the bytes representation. 


**Breaking Changes**
* None


## PR #280 Bug fix: the number of bytes per token were wrongly calculated

This PR fixes the bytes per token calculation.
Generally, we estimate how many bytes are needed to encode the full range of the vocabulary. 
E.g., for a vocab size > 65536, we need 3 bytes for each token in the pbin file. 

The calculation was wrong but coincidentally correct for the GPT2 tokenizer. 



## PR #281: Bug fix: The char-based index is not always consistent with the byte-based index.

The first character of the string "ø This is..." is written on disc as two bytes, namely \xc3\xb8, when encoded as utf-8. 
Therefore, the byte-based index has one more byte/char than the char-based index. 

For consistency, we don't consider any char-based indexes anymore and always refer to byte-based indexes. 


## PR #282: Bug fix: Enforce power of 2 number of bytes per token


Previously, the number of bytes per token was calculated by `math.ceil(log_2(vocab_size)/8)`, leading to ranges between 1 and 4 bytes. 
However, the dataset implementation only support 1, 2 and 4 bytes per token, as defined here

https://github.com/Modalities/modalities/blob/0483362abac93e45850e56adaea7921e96836d59/src/modalities/dataloader/dataset.py#L202-L206

and 

https://github.com/Modalities/modalities/blob/0483362abac93e45850e56adaea7921e96836d59/src/modalities/dataloader/dataset.py#L233-L234

I added a switch case that maps to the respective byte sizes, when packing the data.

This adds some inefficiencies as a vobabulary size > 65536 already requires 4 bytes per token, effectively doubling the storage requirements. 


## PR #283: Bug fix: Only append eod token once when packing / tokenizing

Some HF tokenisers such as `xlm-roberta-large` add special tokens (e.g., eod token) automatically when encoding text, whereas others, such as `gpt2`, do not add special tokens. 

This side-effect in the transformers library has lead to the eod token being appended twice when tokenizing / packing our data. We added a check for this and only append the eod token once now:
https://github.com/Modalities/modalities/blob/1c1ccdc973283c45bc8c9fadf4d20f03e435cd04/src/modalities/dataloader/create_packed_data.py#L327-L330

Additionally, I added a script that verifies the consistency of the indexation and tokenization of a given JSONL file. We run the indexation and tokenization routines in modalities and compare it to tokenized JSONL file to which we applied the HF tokenizer directly. 

## PR #379 Instruction Tuning Support

  * New entry point `apply_chat_template` to form chats and create index and pbin files of it
  * A wrapper for collate functions to include tokens in the loss which appear between indicator tokens
  * A new parameter for the PackedMemMapDatasetContinuous to allow not to re-use the last target token
  * A tutorial how to apply instruction-tuning on a Huggingface Model


## PR #359 Activation Checkpoint with FSDP2 

This PR adds activation checkpointing (AC) support for FSDP2. 
There are now three AC variants: 
* Full AC (same as before, where entire complete modules get ACed, leading to the largest memory footprint reduction)
* Selective Layer AC (only very nth layer or module is ACed)
* Selective OP Ac (only certain OPs, typically low memory but compute intense, are checkpointed)

## PR #374 Tensor Parallelism Support

* adds support for Tensor Parallelism (including Sequence Parallelism). 
* adds a debugging toolkit to track the input and output tensors during a forward pass, gradients during the backward pass and weight tensors.
Tensors can be either normal Tensors or DTensors.  


## PR #389 Benchmark Tooling 
* adds benchmarking tooling to modalities and allows for scaling benchmarks across varying number of nodes and the cartesian product of configurable hyper parameters.

**Breaking Changes**
* Renaming: EvaluationResultToDiscSubscriberConfig.output_path -> EvaluationResultToDiscSubscriberConfig.output_file_path



## PR #410 MFU incorporates dp_degree now instead of world_size

This PR fixes the MFU and throughput calculations by taking the dp degree into account instead of the world size. When we use parallelization strategies on top of FSDP, then the world size is different from the  data parallel degree. This needs to be reflected in throughput and MFU metric calculations, as done by this PR. 

**Breaking Changes**
* Existing configs need to be adapted to correctly use dp degree rather than world size. 


## PR #425 Monitoring improvements
This PR improves training monitoring and logging across runs besides some other changes we did along while testing out scalability.

**General Changes**
* Configurable multi-layer FSDP units
* Option to provide experiment root path to modalities
* Added steppable profiler (e.g., for tracing of forward/backward passes)
* Fix: Hybrid sharding now correctly configurable
* Completely refactored the Profiling 
* Improved error handling. Errors are now captured and stored as JSON
* Add tutorials on Einsum Transformer (Example model integration) and profiling

**Breaking Changes**
* experiments_root_path is now exposed on an API level

## PR #XXX Quality-based document selection and up/downsampling

This PR adds a way to build a training blend by filtering documents on quality signals
and choosing how heavily each dataset is sampled, plus a fast way to see the token
budget a given selection yields before committing to a tokenization run.

**Motivation**

Two kinds of quality signal exist in practice: metrics a corpus already carries in its
own records (`fw_edu_scores`, `proxy_score`, `finemath_scores`, `perplexity`, ...), and
external per-document annotations that have to be joined on. Neither could be used to
shape a blend, and the only way to change a dataset's share was to duplicate it on disk
-- which is what the `_epoch_1` / `_epoch_2` directory convention did.

**General changes**

* New `modalities quality` command group with one subcommand per stage:
  `calibrate`, `build-sidecar`, `join-annotations`, `build-cube`, `preview`, `apply`
  and `write-packing-configs`.
* New `src/modalities/dataloader/preprocessing/quality/` package:
  * `registry` declares each dataset's source and how it joins to annotations. Four key
    kinds are supported, covering corpora that store a plain id, an id wrapped in
    `<urn:uuid:...>`, no id at all (keyed by a hash of the text), and a `<file>/<line>`
    pointer into a separate source corpus.
  * `tokens` measures per-dataset token estimators. Estimates are per document and
    based on the text rather than the stored line, because quality correlates with
    length and several corpora keep multiple renderings of a document in one record.
  * `sidecar` streams each JSONL once and records one row per document: position,
    length, estimated tokens, join key and native metrics.
  * `annotation_join` joins annotations by bucketing both sides on a hash of the key,
    so a split of billions of rows never needs a single hash table. Coverage,
    duplicate keys and unmatched documents are reported rather than hidden.
  * `cube` aggregates a sidecar into grouped document and token counts, which is what
    makes `preview` return in microseconds. Thresholds landing on a bin edge are exact;
    one landing inside a bin is reported as interpolated instead of silently guessed.
  * `selection` evaluates a YAML selection over both annotation labels and native
    metrics, with ordinal scales declared explicitly.
  * `materialize` writes the selection out as filtered `.idx` files.
* New `WeightedCombinedDataset` (component `dataset`/`weighted_combined`), which takes a
  float repeat factor per dataset. A ratio of 2.5 draws a dataset two and a half times
  per epoch and 0.3 draws three tenths of it, without duplicating anything on disk. The
  partial pass is chosen by a seeded affine permutation, so it is deterministic across
  ranks and restarts and spreads across the whole dataset rather than taking a prefix.
* New `TokenizerInstantiationModel`, so a tool can reuse a packing config for its
  tokenizer without also having to satisfy that config's `settings`.

**Notes**

Selection produces a filtered index rather than a filtered copy of the corpus:
`PackedDataGenerator` already tokenizes exactly the documents its index lists, so
`pack_encoded_data` consumes the output unchanged and no packing code was touched. An
ablation therefore costs megabytes of index rather than a second copy of the data, and
the source tree is never written to.

**Breaking Changes**

None. `CombinedDataset` and every existing config keep working as before.


## PR #XXX Quality selection: performance and sharding

Follow-up to the quality-selection PR, from profiling the pipeline against the real
43 TB blend rather than a fixture. Three of the four stages were far slower than they
needed to be, and two of them could not be parallelised at all.

**General changes**

* Native metrics declared as a plain field path (`.fw_edu_scores`,
  `.metadata.dclm_plus2."__label__1"`) are now read by direct dictionary lookup instead
  of jq. `jq.compile(...).input_value(record)` re-serialises the whole document on every
  call, which on a 21 KB record cost more than twenty times the rest of building a
  sidecar row; the full pass measured **29 MB/s with jq against 374 MB/s without, 13x
  end to end**. Patterns jq cannot reduce to a field chain still use jq, and
  `build-sidecar` now warns when one does, since that pattern then dominates the stage.
* `build_cube` groups with Arrow's C++ kernels instead of a Python loop over documents,
  batching row groups so cardinality saturates before each grouping pass:
  **246k -> 2.39M rows/s, 9.9x**. For the full blend that is ~50 minutes rather than
  ~8.5 hours, and the stage was not shardable, so it was a hard floor.
* `build-sidecar` takes `--shard_id`/`--num_shards`. Work is divided per JSONL file
  across every selected dataset, so one array covers the whole blend however unevenly
  the file counts fall. Previously the only split was per dataset, leaving a floor of
  the slowest dataset -- ~71 h for `finepdfs-en`.
* Annotation bucketing moved out of `join-annotations` into its own shardable
  `bucket-annotations` stage. Each task writes its own file per bucket and the join
  reads all of them, so the result is identical to a single-task run. Bucketing 13.9 bn
  annotation rows was ~32 h serial, with a ~8.7 h floor from the largest single split.
* `join-annotations` refuses to run against an incomplete bucketing run rather than
  silently dropping the annotations a missing task was carrying, which would have looked
  exactly like a corpus that was never annotated.

**Notes**

Measured on this cluster: sequential read from `/data` is ~282 MB/s per stream and
~3.8 GB/s aggregate. With the jq fix the sidecar pass is I/O-bound, so beyond ~16
concurrent tasks the storage is the limit rather than the code. End to end the one-time
setup goes from ~88 h to ~4 h on two nodes; previewing a selection is unaffected at
~10 s for the whole blend, because it only ever reads the cubes.

**Breaking Changes**

* `modalities quality build-sidecar` no longer takes `--file_id`; use
  `--shard_id`/`--num_shards`.
* `modalities quality join-annotations` no longer buckets. Run
  `modalities quality bucket-annotations` first. Its `--num_buckets` and
  `--rebuild_buckets` options moved to that command (`--rebuild_buckets` is now
  `--force`).


## PR #XXX Fix: calibration read scaled with file count

`quality calibrate` read a fixed 20,000 lines from *every* file of a dataset to collect a
2,000-document sample, so its cost scaled with the file count rather than the sample size.
Over the real blend that came to **30.3 TB and ~30 hours** -- `dolmino` alone, at 40,003
files, accounted for 26 TB. Reported from a real run that was still going after an hour
having finished 9 of 19 datasets.

**General changes**

* The sampler now draws from at most `max_probe_files` files (default 32), spaced evenly
  across the dataset, reading only enough lines from each to fill the sample. Full
  calibration of all 19 datasets: **4 minutes**, about 1 GB read. The three worst
  datasets together (`dolmino`, `finephrase`, `hplt-de`) now take 67 s.
* `calibration.yaml` is written after each dataset instead of once at the end, so
  interrupting the stage keeps what it already measured. Previously an hour of work was
  discarded on Ctrl-C.

**Notes**

Spread is preserved -- the probe files are spaced across the dataset, not taken from the
front -- and the sample is still trimmed with a seeded choice, so calibration stays
reproducible. Tests cover both: that the number of files opened stays bounded regardless
of dataset size, and that probe files reach both ends of the file list.


## PR #XXX Fix: bucket writer buffered the whole input when bucket count was high

`bucket-annotations` OOM-killed all 64 tasks of a real run at 24 GB each. The writer
flushed a bucket once it held `flush_rows` (100,000) rows, which bounds nothing: spread
50 M rows over 1024 buckets and each holds ~49 k, so no bucket ever reaches the threshold
and the entire input accumulates as Python dicts until the writer closes.

**General changes**

* The cap is now on the **total** rows buffered across all buckets (`max_buffered_rows`,
  default 500,000); reaching it flushes every bucket. Measured on the shard that caused
  the failure -- 50 M rows at 1024 buckets -- peak RSS is **2.36 GB** against the
  previous unbounded growth, at an unchanged 121k rows/s.
* `bucket_annotations` refuses to write into a directory holding output from a run with a
  different `num_shards`. A sharded run cannot clear the directory, so leftovers would
  otherwise be mixed in and a bucket would be read as rows from two incompatible runs.

**Notes**

The completeness guard added earlier did its job here: `join-annotations` refused to run
against the partial buckets the OOMed array left behind ("63 of 64 bucketing tasks
finished, missing shard id 0") rather than silently producing sidecars with no labels.
Without it the failure would have surfaced much later as an unexplained 0 % coverage.

Tests cover the bound directly: 20,000 rows over 1024 buckets with a 1,000-row cap, with
the buffered total asserted after every row, plus that repeated flushes of one bucket
still yield one file with every row.


## PR #XXX Fix: metadata write/read race in bucket-annotations

The guard added in the previous entry -- refusing to mix output from runs with different
array sizes -- read every `_meta.*.json` in a split directory with a bare `json.loads`.
The writer used `Path.write_text`, which truncates before writing, so the file is briefly
empty; sibling tasks of the same array read those files, and 12 of 64 tasks of a real run
died with `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.

**General changes**

* Metadata is written to a per-task `_meta.<shard>.json.tmp` and renamed onto the final
  name. Rename is atomic, so a concurrent reader sees either the old file or the complete
  new one.
* Both readers -- the guard and `read_bucket_metadata` -- skip a file they cannot parse
  rather than propagating the error, and both ignore `*.tmp`. Skipping is the safe
  direction for `read_bucket_metadata`: an unread file leaves its shard id unseen, so the
  run reports as incomplete instead of joining missing annotations. If no file at all can
  be read it now raises a clear error rather than an `AttributeError`.

**Notes**

Verified on the real `finewiki` split: four sequential shards into one directory, 43.1 M
rows, metadata reading back as a complete run with a matching row total and no `.tmp`
left behind. Tests cover a truncated metadata file mid-run, an unreadable file making the
run report incomplete, `.tmp` being ignored, and a thread rewriting metadata while the
guard runs repeatedly.


## PR #XXX Fix: the join re-read the whole annotation split per sidecar part

`join_annotations` looped over sidecar parts on the outside and annotation buckets on the
inside, so the entire bucketed split was read once per part. Because a part's documents
hash across every bucket, that is full re-reads, not partial ones. Measured against the
real blend after bucketing completed:

```
hplt-es        502 parts x 107.3 GB =  53.9 TB
nemotron-cc  5,319 parts x  23.1 GB = 122.7 TB
climbmix-en  6,543 parts x  24.6 GB = 161.3 TB
TOTAL                                454.2 TB  = 451 h of reading
```

**General changes**

* Sidecar parts are processed in batches (`max_batch_keys`, default 20 M documents) and
  each annotation bucket is read once per batch. Read amplification drops from the part
  count to the batch count: **454 TB to 5.9 TB**, 451 h to 5.8 h serial.
* Within a batch each bucket is filtered with `pyarrow.compute.is_in` before anything is
  materialised in Python, so memory is bounded by the batch rather than by the bucket -- a
  bucket of a billion-row split holds millions of rows, of which one batch wants a few
  thousand.
* Bucket file lists are globbed once and cached, saving 65,536 directory scans per batch on
  a 1024-bucket split written by 64 tasks.
* Duplicate annotation keys are now counted once, among the keys the join actually wants.
  The previous figure was inflated by the part count: `finewiki-it` reported 868,586 where
  the real number is 36,747.
* New `3a_join_annotations.sbatch` runs one array task per annotated dataset, resolving the
  dataset from the registry so the mapping cannot drift. Wall time becomes the slowest
  dataset (~2 h) rather than the sum. `3b_build_cubes.sbatch` follows it.

**Notes**

Verified on the real `finewiki-it` sidecar: 1,799,759 of 1,799,759 documents annotated,
100 % coverage, in 48 s against 107 s for a quarter of the documents before. Tests assert
that the batch size does not change any document's label, that a bucket is never read
twice within a batch, and that a duplicate key is counted once rather than once per part.


## PR #XXX Fix: resumable join, and a cube stage that fails clearly

A `nemotron-cc` join reached `5319/5319` parts and was then killed by the 12-hour limit in
`3a_join_annotations.sbatch` during its final write-back, leaving 10 of 5,319 parts without
labels. `build-cube` then died on that inconsistency with
`KeyError: Field "educational_value" does not exist in schema`, after building 9 cubes and
before attempting 6 datasets that were perfectly healthy.

**General changes**

* `join-annotations --resume` skips sidecar parts that already carry the label columns and
  reports the count. Finishing the interrupted run took **9 minutes rather than 12 hours**,
  skipping 5,309 parts. Off by default: resuming after re-bucketing the annotations would
  silently keep the old labels, so continuing has to be asked for explicitly.
  `3a_join_annotations.sbatch` passes it when `JOIN_RESUME` is set.
* `build_cube` reads every part's schema rather than assuming they match the first one's,
  and raises `CubeError` naming the dataset, how many parts are missing which columns, and
  the `--resume` command that finishes the job.
* `build_cubes` no longer aborts the stage on one bad dataset. It builds every healthy one,
  logs the failures together and re-raises so the job still exits non-zero.
* Per-dataset `join_report/<dataset>.json`, plus a merged `join_report.json`. Sixteen
  parallel `--only` tasks had been overwriting one shared file, leaving only the last
  dataset's coverage.
* `3a_join_annotations.sbatch` time limit 12 h -> 48 h, with the measured `nemotron-cc`
  figure recorded next to it.

**Notes**

The join being CPU-bound in Python is why `nemotron-cc` took 12 h where bytes-read implied
2 h. Resolving in Arrow with `pc.index_in` and `Table.take` instead of per-key Python dicts
should be worth 10-50x and is worth doing, but it is a separate change and not needed to
produce a blend.


## PR #XXX Fix: preview refused to admit it was scanning 1.7 bn documents

`evaluate_blend` caught the `SelectionError` a cube raises for a field it was not grouped
on and quietly scanned the per-document sidecar instead. A sidecar scan is exact but reads
every document, so a preview advertised as taking seconds ran for over ten minutes on the
real blend: `nemotron-cc` thresholded `commercial_bias`, which the join attaches but the
cube does not group, and `dolmino` thresholded `dclm_plus2`, which exists only under one of
its subdirectories and was null in all of 400 sampled sidecar parts.

**General changes**

* The fallback is now opt-in via `--allow_fallback`. Without it every unanswerable
  predicate is collected and reported together, naming the dataset, the field, the
  dimensions the cube does carry, and how many documents a scan would read. The real
  selection now fails in **16 s** with both problems named, instead of hanging.
* `build-cube --label_dimension` (repeatable) chooses which annotation columns to group on,
  so a field a selection needs can be added. The flag replaces the default seven rather
  than extending it, and each field multiplies the cell count by its number of levels.
* The example selection no longer thresholds fields the default cubes lack, with a comment
  at each site saying why and how to re-enable it. The registry records that dolmino's
  `dclm_plus2` is confined to `stem-heavy-crawl`.

**Notes**

First full preview of the real blend: 13.7 s across 19 cubes, 3.04 T effective tokens
against a 400 B target. Coverage is 100 % for finewiki, HPLT, Nemotron-CC, ClimbMix and
KletterMix, and 92.5-99.7 % for FinePDFs.


## PR #XXX Fix: pin the source file list a sidecar was built against

A sidecar row locates its document by `(file_id, byte_offset, byte_len)`, and `file_id`
was a position in a file list re-derived from the filesystem at every stage. That makes
every recorded offset depend on the source tree never changing, with nothing recording
what it looked like.

An ongoing transfer then re-sharded four corpora after their sidecars were built.
`Nemotron-CC` went from 606 MB files to 137 MB files while keeping nearly the same file
count, so the count comparison that existed passed and the offsets pointed past end of
file. Eleven of nineteen datasets were unusable and only one -- whose file count fell to
zero -- failed loudly. The rest would have packed a blend of wrong byte ranges.

**General changes**

* `build-sidecar` records its file list in `sidecar/<dataset>/_files.json`, written
  atomically because a sharded build has every task describing the same list.
* `apply` resolves ids through that manifest instead of re-globbing, so a file added to
  the tree cannot renumber anything, and refuses to run if a recorded file changed size or
  disappeared. Paths are relative, so moving or snapshotting a tree stays valid.
* New `quality verify-sidecar`: seeks to recorded offsets and compares the document found
  against the recorded text length, catching a file rewritten at the same size. Rows at
  offset zero are skipped -- the first document of any JSONL file parses, so they succeed
  against a completely different file, which is how the broken sidecars looked healthy.
  `--adopt` stamps a manifest onto a pre-existing sidecar, but only one that verifies.
* `build_sidecars` no longer defaults its index root to the source tree. `SidecarBuilder`
  writes a `.idx` beside each JSONL when given no index root, which would modify a shared
  read-only corpus; it now defaults to `work_dir/idx`.

**Testing**

* `slurm/make_smoke_snapshot.py` freezes ~1 GB of five corpora, chosen to cover all four
  distinct join-key kinds plus the native-metrics-only path -- every branch of the join --
  and `smoke_registry.yaml` / `smoke_selection.yaml` run the full pipeline over it in
  minutes rather than 15 hours.
* `slurm/check_smoke_run.py` compares packed token counts against the preview's estimates,
  loads the output as a `WeightedCombinedDataset` including a fractional repeat factor,
  and asserts nothing was written under the source root.
* `test_file_manifest.py` covers the re-shard that preserves the file count, a prepended
  file that would renumber ids, a removed file, `apply` refusing a drifted tree, and that
  adoption is refused for a sidecar that does not verify.


## PR #XXX Fix: a resumed join reported 0% coverage

`--resume` counted only the parts it re-joined, so a run that skipped everything wrote a
report saying 0 documents and 0.0 coverage. On the smoke run it overwrote three datasets'
genuine coverage figures with zeros, which reads as a failed join rather than a skipped
one. Coverage is a property of the sidecar, not of the run, so skipped parts now
contribute their existing labels to the totals, and `n_parts_resumed` records how many
were not redone.


## PR #XXX Fix: the token estimator was 16-19% out, and looked stable while being wrong

The end-to-end smoke run compared the preview's estimates against a real packing run for
the first time. Document counts matched exactly, but `finewiki-de` tokens were 10.7% out,
and chasing that found two independent defects in the calibration.

**The sample was a prefix.** `_sample_documents` took the first N documents of each probe
file. Being deterministic, it produced identical ratios across every seed, which reads as
stability rather than as a sample that never moves. On the FineWiki snapshot the first
2,000 documents gave 3.531 bytes per token where the whole file gives 4.214 -- 16% out,
applied to every token figure downstream. I introduced this when bounding an earlier 30 TB
read: I capped the read by taking a prefix and never made the within-file sample spread.

**One global ratio cannot describe a corpus.** FineWiki's ratio runs from 3.571 for
documents under a kilobyte to 34.648 for the nine documents above 256 KB -- and those nine
hold 5.7% of all bytes. A single sum ratio is therefore hostage to whether the sample
caught them, which is why the global estimator's error swung between -19.4% and +4.2%
across seeds.

**General changes**

* Documents are sampled at offsets spread evenly across each file, and the document
  *containing* each offset is taken rather than the one following it. That makes selection
  proportional to length, which is what a byte-weighted ratio needs: the top stratum went
  from 2 sampled documents per 2,000 to 44.
* The ratio is measured per size stratum (log-spaced, six of them) and applied per document
  from the length the sidecar already records exactly. A stratum reached by fewer than 20
  documents falls back to the corpus-wide ratio rather than becoming an estimator of its
  own.
* The corpus-wide ratio now uses inverse-probability weights. Under length-proportional
  sampling a plain sum ratio is weighted by the square of length and came out 62% low.
* Each document's ratio is measured on a slice of at most 64 KB, taken from a random
  position inside it. Length-proportional sampling means the multi-megabyte documents do get
  sampled, and tokenizing them in full took calibration from 1.5 minutes to over 10; a
  document's ratio is far more uniform within itself than across the corpus, so a slice
  measures it well. Calibration is now ~15 s per dataset.
* `--sample_size` default raised from 2000 to 4000, for margin.

**Result on the FineWiki snapshot**, against the true token count of all 27,846 documents:

| estimator | worst error over 5 seeds |
|---|---|
| global ratio, prefix sample (before) | 16.2% bias, invisible across seeds |
| global ratio, spread sample | 19.4% |
| stratified, spread sample | 0.8% |

**Notes**

Calibrations written before this change have no strata and fall back to the global ratio,
so they still load; they should be re-measured. Changing a calibration changes `est_tokens`
in the sidecars, which means rebuilding them -- worth folding into the rebuild the source
re-shard already forces.


## PR #XXX End-to-end validation of the estimator against a real packing run

Re-ran the whole smoke pipeline after the estimator fix. Estimated against packed tokens,
with document counts as the exact control:

| dataset | est tokens | packed | error | docs selected | docs packed |
|---|---|---|---|---|---|
| finewiki-de | 18,506,144 | 18,438,976 | -0.36% | 20,693 | 20,693 |
| finepdfs-es | 8,644,194 | 8,505,597 | -1.60% | 825 | 825 |
| climbmix-en | 54,527,783 | 54,493,399 | -0.06% | 60,834 | 60,834 |
| klettermix-de | 19,799,809 | 19,818,254 | +0.09% | 22,915 | 22,915 |
| dolmino | 48,320,010 | 48,508,512 | +0.39% | 8,106 | 8,106 |
| **total** | **149,797,940** | **149,764,738** | **-0.02%** | | |

`finewiki-de` was -10.72% before the fix. Document counts match exactly, which is the
stronger check: the filtered index names exactly the selected documents, so any difference
would be a defect in materialize rather than estimator error.

The blend also loads: 8 packed files combined through `WeightedCombinedDataset` with repeat
factors 0.5/1.0/1.5/2.0, length 74,033 against 74,032 expected, samples pulled at both
boundaries and the middle. The fractional factors exercise the partial-pass permutation,
which no test on real data had reached. Nothing was written under the source root.


## PR #XXX Perf: resolve the join in Arrow, and drop bucket routing

The join built a `dict[key, list[(part, row)]]` over every document, a per-row dict of
labels for every document, and a Python list comprehension per label column -- a handful of
Python objects per document, at 1.7 bn documents for Nemotron-CC. Each part is now resolved
with one `index_in` against the batch's lookup table and one `take` per label column, and
the key column stays Arrow in the outer loop, where materialising it had been 1.7 bn Python
strings before any joining began.

Profiling what remained showed the next cost was not per-row work at all: 22 s of reads and
14 s of a thousand separate `is_in` calls, out of 47 s, because a 554 MB split is
partitioned into 1024 files of roughly 540 KB. The routing those buckets exist for also
turned out to decide nothing -- a batch holds millions of keys, which hash across every
bucket, so the profile recorded all 1024 files being read anyway, after a blake2b call per
key in Python to choose them. Routing is gone, replaced by one `pyarrow.dataset` scan with
the key filter pushed into it. Arrow applies the filter per row group and reads in parallel,
so memory stays bounded by matching rows rather than by split size.

**Measured** on `finewiki-en`, 6.6 M documents against the 43 M-row FineWiki split, both
implementations on separate copies of the same sidecar:

| | elapsed | per document |
|---|---|---|
| before | 88.7 s | 13.41 us |
| after | 42.1 s | 6.36 us |

Equivalence was verified twice rather than assumed: 79,375,860 label values identical across
12 columns on the benchmark, and 1,441,584 values identical across all four join-key kinds
on the smoke blend, with duplicate-key counts matching exactly (390 / 2 / 1,936 / 6,025).
That check mattered because "keep the first row seen" had to survive reimplementation as
`index_in` over unique values, which reports each value's first position.

**What this does not fix, and it is the larger cost.** The annotation split is scanned once
per batch, and batch count scales with documents:

| dataset | documents | batches @ 20 M | annotation rows | row scans |
|---|---|---|---|---|
| finewiki-en | 6.6 M | 1 | 43 M | 43 M |
| hplt-de | 176 M | 9 | 3.76 bn | 33.8 bn |
| climbmix-en | 553 M | 28 | 552 M | 15.5 bn |
| nemotron-cc | 1.70 bn | 85 | 747 M | 63.5 bn |

So `nemotron-cc` reads its 22 GB split 85 times, about 1.9 TB, and that is what its twelve
hours were mostly spent on. The benchmark above has exactly one batch, so it measures the
overhead this commit addresses and none of the re-scanning, and it should not be read as a
prediction for `nemotron-cc`.

Reducing the scans means letting batches hold far more keys, which today is bounded by the
batch holding a full `pa.Table` per part. Collecting only key columns for the scan and
re-reading the parts to write back would cut that by roughly an order of magnitude, at the
cost of reading the sidecar twice -- cheap against 85 scans of the split. Not attempted
here; it changes the memory profile of a stage that has already been OOM-killed once.
